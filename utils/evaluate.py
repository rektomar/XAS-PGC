import torch
import numpy as np
import pandas as pd
import networkx as nx

from fcd_torch import FCD
from rdkit import Chem
from utils.molecular import mols2gs, gs2mols, mols2smls, get_vmols 
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.stats import entropy, gaussian_kde
from utils.props import calculate_props_df
from eden.graph import vectorize


# (So far) requires additional installation of:
#   Cython (for eden) - pip install cython
#   eden - pip install git+https://github.com/fabriziocosta/EDeN.git
#   scikit-learn - pip install scikit-learn

def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))

        nx_graphs.append(G)
    return nx_graphs

def kernel_compute(X, Y=None, n_jobs=None):
    X = vectorize(X, complexity=4, discrete=True)
    if Y is not None:
        Y = vectorize(Y, complexity=4, discrete=True)
    return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)

### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
def compute_nspdk_mmd(samples1, samples2, n_jobs=None):

    X = kernel_compute(samples1,             n_jobs=n_jobs)
    Y = kernel_compute(samples2,             n_jobs=n_jobs)
    Z = kernel_compute(samples1, Y=samples2, n_jobs=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)

##### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py
def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, n_jobs=20)
    return mmd_dist

def continuous_kldiv(X_baseline, X_sampled) -> float:
    # taken from https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/chemistry.py
    kde_P = gaussian_kde(X_baseline)
    kde_Q = gaussian_kde(X_sampled)
    x_eval = np.linspace(np.hstack([X_baseline, X_sampled]).min(), np.hstack([X_baseline, X_sampled]).max(), num=1000)
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10

    return entropy(P, Q)

def discrete_kldiv(X_baseline, X_sampled) -> float:
    P, bin_edges = np.histogram(X_baseline, bins=10, density=True)
    P += 1e-10 

    Q, _ = np.histogram(X_sampled, bins=bin_edges, density=True)
    Q += 1e-10

    return entropy(P, Q) 

def prop_kldiv(props_gen: pd.DataFrame, props_ref: pd.DataFrame):
    kldivs = {}
    for p in ['BCT', 'logP', 'MW', 'TPSA']:
        kldiv = continuous_kldiv(X_baseline=props_ref[p], X_sampled=props_gen[p])
        kldivs[p] = kldiv

    for p in ['numHBA', 'numHBD', 'numRB', 'numAlR', 'numArR']:
        kldiv = discrete_kldiv(X_baseline=props_ref[p], X_sampled=props_gen[p])
        kldivs[p] = kldiv

    # missing internal pairwise similarities

    partial_scores = [np.exp(-score) for score in kldivs.values()]
    score = np.sum(partial_scores) / len(partial_scores)

    return score


def metric_v(vmols, num_mols):
    return len(vmols) / num_mols

def metric_u(vsmls, num_mols):
    num_umols = len(list(set(vsmls)))
    if num_umols == 0:
        return 0., 0.
    else:
        return num_umols / len(vsmls), num_umols / num_mols

def metric_n(vsmls, tsmls, num_mols):
    usmls = list(set(vsmls))
    num_umols = len(usmls)
    if num_umols == 0:
        return 0., 0.
    else:
        num_nmols = num_umols - sum([1 for mol in usmls if mol in tsmls])
        return num_nmols / num_umols, num_nmols / num_mols

def metric_m(ratio_v, ratio_u, ratio_n):
    return ratio_v*ratio_u*ratio_n

def metric_s(mols, num_mols):
    mols_stable = 0
    bond_stable = 0
    sum_atoms = 0

    for mol in mols:
        mol = Chem.AddHs(mol, explicitOnly=True)
        num_atoms = mol.GetNumAtoms()
        num_stable_bonds = 0
        for atom in mol.GetAtoms():
            num_stable_bonds += int(atom.HasValenceViolation() == False)

        mols_stable += int(num_stable_bonds == num_atoms)
        bond_stable += num_stable_bonds
        sum_atoms += num_atoms

    return mols_stable / float(num_mols), bond_stable / float(sum_atoms)

def metric_f(smls_gen, smls_ref, device="cuda", canonical=True):
    if len(smls_gen) < 2 or len(smls_ref) < 2:
        return np.nan
    fcd = FCD(device=device, n_jobs=2, canonize=canonical)
    return fcd(smls_ref, smls_gen)

def metric_k(smls_gen, smls_ref):
    if len(smls_gen) < 2 or len(smls_ref) < 2:
        return np.nan
    mols_gen = [Chem.MolFromSmiles(s) for s in smls_gen]
    mols_ref = [Chem.MolFromSmiles(s) for s in smls_ref]

    props_gen = calculate_props_df(mols_gen)
    props_ref = calculate_props_df(mols_ref)
    return prop_kldiv(props_gen, props_ref)

def metric_nspdk(smls_gen, smls_ref):
    if len(smls_gen) < 2 or len(smls_ref) < 2:
        return np.nan
    mols_gen = [Chem.MolFromSmiles(s) for s in smls_gen]
    mols_ref = [Chem.MolFromSmiles(s) for s in smls_ref]

    graphs_gen = mols_to_nx(mols_gen)
    graphs_ref = mols_to_nx(mols_ref)

    return nspdk_stats(graphs_ref, graphs_gen)


def evaluate_molecules(
        x,
        a,
        loaders,
        atom_list,
        evaluate_trn=False,
        evaluate_val=False,
        evaluate_tst=False,
        metrics_only=False,
        canonical=True,
        preffix='',
        device="cuda"
    ):
    num_mols = len(x)

    mols = gs2mols(x, a, atom_list)
    smls = mols2smls(mols, canonical)
    vmols, vsmls = get_vmols(smls)

    ratio_v = metric_v(vmols, num_mols)
    ratio_u, ratio_u_abs = metric_u(vsmls, num_mols)
    ratio_n, ratio_n_abs = metric_n(vsmls, loaders['smiles_trn'], num_mols)
    ratio_s = metric_m(ratio_v, ratio_u, ratio_n)
    ratio_m, ratio_a = metric_s(mols, num_mols)

    metrics = {
        f'{preffix}valid': ratio_v,
        f'{preffix}unique': ratio_u,
        f'{preffix}unique_abs': ratio_u_abs,
        f'{preffix}novel': ratio_n,
        f'{preffix}novel_abs': ratio_n_abs,
        f'{preffix}score': ratio_s,
        f'{preffix}m_stab': ratio_m,
        f'{preffix}a_stab': ratio_a
    }

    if evaluate_trn == True:
        metrics = metrics | {
            f'{preffix}fcd_trn'  : metric_f(vsmls, loaders['smiles_trn'], device, canonical),
            f'{preffix}kldiv_trn': metric_k(vsmls, loaders['smiles_trn']),
            # f'{preffix}nspdk_trn': metric_nspdk(vsmls, loaders['smiles_trn']),
        }
    if evaluate_val == True:
        metrics = metrics | {
            f'{preffix}fcd_val'  : metric_f(vsmls, loaders['smiles_val'], device, canonical),
            f'{preffix}kldiv_val': metric_k(vsmls, loaders['smiles_val']),
            f'{preffix}nspdk_val': metric_nspdk(vsmls, loaders['smiles_val']),
        }
    if evaluate_tst == True:
        metrics = metrics | {
            f'{preffix}fcd_tst'  : metric_f(vsmls, loaders['smiles_tst'], device, canonical),
            f'{preffix}kldiv_tst': metric_k(vsmls, loaders['smiles_tst']),
            f'{preffix}nspdk_tst': metric_nspdk(vsmls, loaders['smiles_tst']),
        }

    if metrics_only == True:
        return metrics
    else:
        return vmols, vsmls, metrics

def print_metrics(metrics):
    return f'v={metrics["valid"]:.2f}, ' + \
           f'u={metrics["unique"]:.2f}, ' + \
           f'n={metrics["novel"]:.2f}, ' + \
           f's={metrics["score"]:.2f}, ' + \
           f'ms={metrics["m_stab"]:.2f}, ' + \
           f'as={metrics["a_stab"]:.2f}'

def resample_invalid_mols(model, num_samples, atom_list, max_atoms, canonical=True, max_attempts=10):
    n = num_samples
    mols = []

    for _ in range(max_attempts):
        x, a = model.sample(n)
        vmols, _ = get_vmols(mols2smls(gs2mols(x, a, atom_list), canonical))
        mols.extend(vmols)
        n = num_samples - len(mols)
        if len(mols) == num_samples:
            break

    x_valid, a_valid = mols2gs(mols, max_atoms, atom_list)
    if n > 0:
        x_maybe, a_maybe = model.sample(n)
        return torch.cat((x_valid, x_maybe)), torch.cat((a_valid, a_maybe))
    else:
        return x_valid, a_valid

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_metrics(gsmls, tsmls, max_atom, atom_list, disrupt_mol=None):
    gmols = [Chem.MolFromSmiles(sml) for sml in gsmls]
    [Chem.Kekulize(mol) for mol in gmols]
    x, a = mols2gs(gmols, max_atom, atom_list)
    if disrupt_mol is not None:
        n, c, i, j = disrupt_mol
        a[n, :, i, j] = 0
        a[n, c, i, j] = 1

    metrics = evaluate_molecules(x, a, tsmls, atom_list, correct_mols=False, metrics_only=True)

    print_metrics(metrics['valid'],
                  metrics['novel'],
                  metrics['unique'],
                  metrics['score'],
                  metrics['novel_abs'],
                  metrics['unique_abs'], abs=True)


if __name__ == '__main__':
    # 10 samples from the QM9 dataset
    max_atom = 9
    atom_list = [0, 6, 7, 8, 9]
    tsmls = [
            'CCC1(C)CN1C(C)=O',
            'O=CC1=COC(=O)N=C1',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]


    gsmls = [
            'CC1(C)CN1C(C)=O',
            'O=CC1=COC(=O)N=C1',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]
    test_metrics(gsmls, tsmls, max_atom, atom_list, [0, 2, 1, 0])
    test_metrics(gsmls, tsmls, max_atom, atom_list, [4, 2, 1, 0])


    gsmls = [
            'CCC1(C)CN1C(C)=O',
            'CCC1(C)CN1C(C)=O',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]
    test_metrics(gsmls, tsmls, max_atom, atom_list)


    gsmls = [
            'CC1(C)CN1C(C)=O',
            'O=CC1=COC(=O)N=C1',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]
    test_metrics(gsmls, tsmls, max_atom, atom_list)


    smls_1 = [
        'CCC1(C)CN1C(C)=O',
        'O=CC1=COC(=O)N=C1',
        'O=CC1(C=O)CN=CO1',
    ]

    smls_2 = [
        'CC1(C)CN1C(C)=O',
        'O=CC1=COC(=O)N=C1',
        'O=CC1(C=O)CN=CO1'
        ]

    nspdk = metric_nspdk(smls_1, smls_2)
    print(f'NSPDK score: {nspdk}')

    fcd = metric_f(smls_1, smls_2)
    print(f'fcd: {fcd}')

    kldiv = metric_k(smls_1, smls_2)
    print(f'KLdiv score: {kldiv}')

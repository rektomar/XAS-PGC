import os
import torch
import pandas as pd

from rdkit import Chem
from fcd_torch import FCD
from utils.molecular import mols2gs, gs2mols, mols2smls, get_vmols


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
    fcd = FCD(device=device, n_jobs=2, canonize=canonical)
    return fcd(smls_ref, smls_gen)

def evaluate_molecules(
        x,
        a,
        loaders,
        atom_list,
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

    if evaluate_val == True:
        metrics = metrics | {
            f'{preffix}fcd_val': metric_f(vsmls, loaders['smiles_val'], device, canonical),
            f'{preffix}nspdk_val': 0
        }
    if evaluate_tst == True:
        metrics = metrics | {
            f'{preffix}fcd_tst': metric_f(vsmls, loaders['smiles_tst'], device, canonical),
            f'{preffix}nspdk_tst': 0
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

def best_model(path):
    files = os.listdir(path)
    dfs = []
    for f in files:
        data = pd.read_csv(path + f)
        data['file_path'] = f.replace('.csv', '.pt')
        dfs.append(data)
    df = pd.concat(dfs, ignore_index=True)
    idx = df['nll_tst_approx'].idxmin()
    return df.loc[idx]['file_path'], df.loc[idx]['nll_val_approx']


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

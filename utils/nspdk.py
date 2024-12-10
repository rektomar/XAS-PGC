import numpy as np
import networkx as nx

from sklearn.metrics.pairwise import pairwise_kernels
from eden.graph import vectorize

from rdkit import Chem

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


def metric_nspdk(smls_gen, smls_ref):
    mols_gen = [Chem.MolFromSmiles(s) for s in smls_gen]
    mols_ref = [Chem.MolFromSmiles(s) for s in smls_ref]

    graphs_gen = mols_to_nx(mols_gen)
    graphs_ref = mols_to_nx(mols_ref)

    return nspdk_stats(graphs_ref, graphs_gen)


if __name__ == "__main__":

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
    

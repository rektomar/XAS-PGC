import numpy as np
import pandas as pd

from rdkit import Chem
from scipy.stats import entropy, gaussian_kde
from utils.props import calculate_props_df


def continuous_kldiv(X_baseline, X_sampled) -> float:
    # taken from https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/chemistry.py
    kde_P = gaussian_kde(X_baseline)
    kde_Q = gaussian_kde(X_sampled)
    x_eval = np.linspace(np.hstack([X_baseline, X_sampled]).min(), np.hstack([X_baseline, X_sampled]).max(), num=1000)
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10

    return entropy(P, Q)

# def discrete_kldiv(X_baseline, X_sampled) -> float:
#     # taken from https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/chemistry.py
#     P, bins = histogram(X_baseline, bins=10, density=True)
#     P += 1e-10
#     Q, _ = histogram(X_sampled, bins=bins, density=True)
#     Q += 1e-10

#     return entropy(P, Q)

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

def metric_k(smls_gen, smls_ref):
    mols_gen = [Chem.MolFromSmiles(s) for s in smls_gen]
    mols_ref = [Chem.MolFromSmiles(s) for s in smls_ref]

    props_gen = calculate_props_df(mols_gen)
    props_ref = calculate_props_df(mols_ref)
    return prop_kldiv(props_gen, props_ref)


if __name__ == '__main__':
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

    kldiv = metric_k(smls_1, smls_2)
    print(f'KLdiv score: {kldiv}')

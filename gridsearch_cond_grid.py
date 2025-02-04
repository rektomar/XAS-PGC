import os
import numpy as np
import torch
import pandas as pd

from pylatex import Document, Package, NoEscape

from utils.datasets import MOLECULAR_DATASETS
from utils.conditional import create_conditional_grid, evaluate_conditional
from utils.plot import plot_grid_conditional, plot_grid_unconditional

from rdkit import Chem, rdBase
rdBase.DisableLog("rdApp.error")


PATT_CONFIG = {
    'qm9': ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'N1C=CC=C1'],
    'zinc250k': ['NS(=O)C1=CC=CC=C1', 'CC(=O)C1=CC=CC=C1O', 'CNC(C)=O', 'O=C1C=CC=CN1CC1=CC=C(F)C=C1', 'COC1=CC=C(F)C2=NC(N)=CC=C12']
}

def find_best(evaluation_dir, dataset, model):
    path = evaluation_dir + f'metrics/{dataset}/{model}/'

    def include_name(name):
        if 'ptree' in name and 'bft' in name:
            return True
        else:
            return False

    b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path) if include_name(f)], ignore_index=True)
    f_frame = b_frame.loc[b_frame['sam_fcd_val'].idxmin()]
    return f_frame['model_path']

def create_grid(path_model, dataset, num_to_show=8, num_to_sample=2000, seed=0, chunk_size=500, useSVG=False):
    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']
    model = torch.load(path_model, weights_only=False)

    patt_smls = PATT_CONFIG[dataset]
    cond_smls = create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, max_atoms, atom_list, seed=seed, chunk_size=chunk_size)

    plot_grid_conditional(cond_smls, patt_smls, fname=f"{dataset}_cond_mols", useSVG=useSVG)


if __name__ == '__main__':
    evaluation_dir = '/mnt/data/density_learning/molspn/gs0/eval/'

    # path_model_qm9 = find_best(evaluation_dir, 'qm9', 'marg_sort')
    # print(path_model_qm9)
    # create_grid(path_model_qm9, 'qm9', useSVG=True)

    path_model_zinc250k = find_best(evaluation_dir, 'zinc250k', 'marg_sort')
    print(path_model_zinc250k)
    create_grid(path_model_zinc250k, 'zinc250k', useSVG=True)



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
    'qm9': ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'C1=CC=CC=C1', 'C1CN1C', 'N1C=CC=C1', 'COC'],
    'zinc250k': ['CC(=O)c1ccccc1O', 'CNC(C)=O', 'Fc1ccc(Cn2ccccc2=O)cc1', 'COc1ccc(F)c2nc(N)ccc12']#, 'FC(F)(F)CN(CC1CC1)C(=O)c1cc[nH]c1']
}

def find_best(evaluation_dir, dataset, model):
    path = evaluation_dir + f'metrics/{dataset}/{model}/'
    exclude = 'ctree'
    b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path) if exclude not in f], ignore_index=True)
    f_frame = b_frame.loc[b_frame['sam_valid'].idxmax()]
    return f_frame['model_path']

def create_grid(path_model, dataset, num_to_show=8, num_to_sample=200, seed=1):
    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']
    model = torch.load(path_model, weights_only=False)

    torch.manual_seed(seed)
    patt_smls = PATT_CONFIG[dataset]
    cond_smls = create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, max_atoms, atom_list)

    plot_grid_conditional(cond_smls, patt_smls, fname=f"{dataset}_cond_mols", useSVG=False)


if __name__ == '__main__':
    evaluation_dir = '/mnt/data/density_learning/molspn/gs0/eval/'
    path_model_qm9 = find_best(evaluation_dir, 'qm9', 'marg_sort')
    create_grid(path_model_qm9, 'qm9')

    path_model_zinc250k = find_best(evaluation_dir, 'zinc250k', 'marg_sort')
    create_grid(path_model_zinc250k, 'zinc250k')
    



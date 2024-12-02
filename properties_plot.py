import torch
import pandas as pd
import numpy as np

from scipy.stats import gaussian_kde

from utils.latex import *
from utils.molecular import gs2mols, isvalid
from utils.evaluate import resample_invalid_mols
from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.props import calculate_props

from rdkit import rdBase
rdBase.DisableLog("rdApp.error")


model_path_config = {
    'qm9': '/home/rektomar/projects/MolSPN/results/training/model_checkpoint/qm9/zero_sort/dataset=qm9_model=zero_sort_order=rcm_nc=100_backend=ptree_nr=40_xnl=3_xns=40_xni=40_anl=5_ans=40_ani=40_device=cuda_lr=0.05_betas=[0.9, 0.82]_num_epochs=40_batch_size=1000_seed=0.pt',
    'zinc250k': '/home/rektomar/projects/MolSPN/results/training/model_checkpoint/zinc250k/zero_sort/dataset=zinc250k_model=zero_sort_order=canonical_nc=100_nr=20_backend=ptree_xnl=4_xns=20_xni=20_anl=6_ans=20_ani=20_device=cuda_lr=0.05_betas=[0.9, 0.82]_num_epochs=40_batch_size=256_seed=0.pt'
}

def calculate_props_all(mols):
    props = []
    for mol in mols:
        try:
            props.append(calculate_props(mol))
        except ZeroDivisionError:  # Zero division problem with SA calculation for really simple molecules
            pass
    return pd.DataFrame(props)

def plot_kde(y, label, y_min, y_max, n_points: int=200):
    kde = gaussian_kde(y)

    y = np.linspace(y_min, y_max, n_points)
    pdf = kde.pdf(y)
    return create_line_plot(y, pdf, label)

def get_lim(x, mode='max'):
    if   mode == 'max':
        lb, ub = x.min(), x.max()
    elif mode == '90pct':
        lb, ub = x.quantile(0.05), x.quantile(0.95)
    return lb, ub

def plot_props(trn_props, gen_props, dataset_name):
    
    for prop in ['SA', 'MW', 'logP', 'QED']:
        prop_min, prop_max = get_lim(trn_props[prop])
        trn_plt = plot_kde(trn_props[prop], 'train set', prop_min, prop_max)
        gen_plt = plot_kde(gen_props[prop], 'generated', prop_min, prop_max)
        create_latex_pgf_plot([trn_plt, gen_plt], prop, 'density (-)', f'{prop} distribution',
                              pdf_filename=f'{dataset_name}_{prop}')

if __name__ == '__main__':
    dataset = 'zinc250k'
    num_samples = 1000
    # load_dataset
    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']
    max_types = MOLECULAR_DATASETS[dataset]['max_types']

    loader_trn, loader_val = load_dataset(dataset, 256, split=[0.8, 0.2], order='canonical')

    # load model
    model_path = model_path_config[dataset]
    model = torch.load(model_path, weights_only=False)
    torch.manual_seed(0)

    x, a = resample_invalid_mols(model, num_samples, atom_list, max_atoms)
    mols = gs2mols(x, a, atom_list)
    mols_valid = list(filter(isvalid, mols))
    gen_props = calculate_props_all(mols_valid)

    # TODO: get dataset in a proper way
    trn_props = pd.DataFrame(loader_trn.dataset.dataset[:])

    plot_props(trn_props, gen_props, dataset)


import torch
from utils.datasets import MOLECULAR_DATASETS

from utils.conditional import create_conditional_grid, evaluate_conditional
from utils.plot import plot_grid_conditional, plot_grid_unconditional

from rdkit import rdBase
rdBase.DisableLog("rdApp.error")

# nice utility for molecule drawings https://www.rcsb.org/chemical-sketch
# preselected pattern smiles for each dataset
patt_grid_config = {
    'qm9': ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'C1=CC=CC=C1', 'C1CN1C', 'N1C=CC=C1', 'COC'],
    'zinc260k': []
}

patt_eval_config = {
    'qm9': ['COC', 'N1NO1'],
    'zinc260k': []
}

model_path_config = {
    'qm9': "results/training/model_checkpoint/qm9/molspn_zero_sort_feat/dataset=qm9_model=molspn_zero_sort_feat_nc=100_nd_n=9_nk_n=5_nk_e=4_nl_n=2_nl_e=2_nr_n=100_nr_e=100_ns_n=20_ns_e=20_ni_n=50_ni_e=50_dc_n=0.6_dc_e=0.6_regime=cat_device=cuda_lr=0.1_num_epochs=1000_batch_size=256_seed=0.pt",
    'zinc250k': ""
}

if __name__ == "__main__":
    dataset = 'qm9'
    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']
    max_types = MOLECULAR_DATASETS[dataset]['max_types']

    # load trained model
    model_path = model_path_config[dataset]
    model = torch.load(model_path, weights_only=False)
    torch.manual_seed(2)

    num_to_sample = 20
    num_to_show = 8  # assuming at least num_to_show of num_samples are valid

    patt_smls = patt_grid_config[dataset]
    cond_smls = create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, max_atoms, atom_list)
    
    # conditional and unconditional sampling grid plots
    plot_grid_conditional(cond_smls, patt_smls, fname=f"{dataset}_cond_mols", useSVG=False)
    plot_grid_unconditional(model, 8, 8, max_atoms, atom_list, fname=f"{dataset}_unco_mols", useSVG=False)

    # conditional sampling metrics eval
    for patt_sml in patt_eval_config[dataset]:
        evaluate_conditional(model, patt_sml, dataset, max_atoms, atom_list, num_samples=1000)

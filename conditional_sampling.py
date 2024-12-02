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
    'zinc250k': ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'C1=CC=CC=C1', 'C1CN1C', 'N1C=CC=C1', 'COC']
}

patt_eval_config = {
    'qm9': ['COC', 'N1NO1'],
    'zinc250k': ['COC', 'N1NO1']
}

model_path_config = {
    'qm9': '/home/rektomar/projects/MolSPN/results/training/model_checkpoint/qm9/zero_sort/dataset=qm9_model=zero_sort_order=rcm_nc=100_backend=ptree_nr=40_xnl=3_xns=40_xni=40_anl=5_ans=40_ani=40_device=cuda_lr=0.05_betas=[0.9, 0.82]_num_epochs=40_batch_size=1000_seed=0.pt',
    'zinc250k': '/home/rektomar/projects/MolSPN/results/training/model_checkpoint/zinc250k/zero_sort/dataset=zinc250k_model=zero_sort_order=canonical_nc=100_nr=20_backend=ptree_xnl=4_xns=20_xni=20_anl=6_ans=20_ani=20_device=cuda_lr=0.05_betas=[0.9, 0.82]_num_epochs=40_batch_size=256_seed=0.pt'
}

if __name__ == "__main__":
    dataset = 'zinc250k'
    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']
    max_types = MOLECULAR_DATASETS[dataset]['max_types']

    # load trained model
    model_path = model_path_config[dataset]
    model = torch.load(model_path, weights_only=False)
    torch.manual_seed(2)

    num_to_sample = 50
    num_to_show = 8  # assuming at least num_to_show of num_samples are valid

    patt_smls = patt_grid_config[dataset]
    cond_smls = create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, max_atoms, atom_list)
    
    # conditional and unconditional sampling grid plots
    plot_grid_conditional(cond_smls, patt_smls, fname=f"{dataset}_cond_mols", useSVG=False)
    plot_grid_unconditional(model, 8, 8, max_atoms, atom_list, fname=f"{dataset}_unco_mols", useSVG=False)

    # conditional sampling metrics eval
    for patt_sml in patt_eval_config[dataset]:
        evaluate_conditional(model, patt_sml, dataset, max_atoms, atom_list, num_samples=1000)

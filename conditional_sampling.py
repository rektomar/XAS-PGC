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
    'qm9': 'COC',
    'zinc260k': []
}

if __name__ == "__main__":
    dataset = 'qm9'
    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']
    max_types = MOLECULAR_DATASETS[dataset]['max_types']

    # trained model path
    model_path = "/home/rektomar/projects/MolSPN/results/training/model_checkpoint/qm9/molspn_vaef_sort/dataset=qm9_model=molspn_vaef_sort_nx=9_nx_x=5_nx_a=4_nz=9_nz_x=5_nz_a=4_nz_y=0_h_x=2048_h_a=1024_h_y=512_l_x=8_l_a=8_l_y=4_l_b=4_ftype=none_device=cuda_lr=0.0001_weight_decay=0.0_num_epochs=100_batch_size=256_seed=0.pt"
    model = torch.load(model_path, weights_only=False)
    torch.manual_seed(2)

    num_to_sample = 1000
    num_to_show = 8  # assuming at least num_to_show of num_samples are valid

    patt_smls = patt_grid_config[dataset]
    cond_smls = create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, max_atoms, atom_list)
    
    plot_grid_conditional(cond_smls, patt_smls, fname=f"{dataset}_cond_mols", useSVG=False)
    plot_grid_unconditional(model, 8, 8, max_atoms, atom_list, fname=f"{dataset}_unco_mols", useSVG=False)

    patt_sml = patt_eval_config[dataset] 
    evaluate_conditional(model, patt_sml, dataset, max_atoms, atom_list, num_samples=num_to_sample)

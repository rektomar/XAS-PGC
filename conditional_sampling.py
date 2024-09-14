import torch
from utils.datasets import MOLECULAR_DATASETS
from utils.molecular import isvalid, mol2g, gs2mols
from utils.plot import grid_conditional, grid_unconditional
from utils.graphs import flatten_graph, unflatt_graph
from models.utils import ohe2cat, cat2ohe

from rdkit import Chem, rdBase
rdBase.DisableLog("rdApp.error")


def create_mask(max_mol_size, submol_size, device='cuda'):
    """Create marginalization masks 'mx' and 'ma' for a single molecule.
       E.g., mx[j] = 1 means j-th node is observed,
             mx[j] = 0 means j-th node is marginalized out,
    Parameters:
        max_mol_size (int): maximal number of atoms in molecule
        submol_size  (int): number of atoms of masked submolecule 
                            submol_size < max_mol_size
        device       (str): target device of new mask
    Returns:
        mx (torch.Tensor): node feature mask     [1, max_mol_size]
        ma (torch.Tensor): adjacency tensor mask [1, max_mol_size, max_mol_size]
    """
    assert submol_size < max_mol_size
    BS = 1
    mx = torch.zeros(BS, max_mol_size,               dtype=torch.bool, device=device)
    ma = torch.zeros(BS, max_mol_size, max_mol_size, dtype=torch.bool, device=device)
    mx[:, :submol_size              ] = True
    ma[:, :submol_size, :submol_size] = True
    return mx, ma

def conditional_sample(model, x, a, submol_size, num_samples, max_mol_size, atom_list):
    """Conditionaly generate a molecule given some other (smaller) molecule.
    Parameters:
        model (torch.nn.Module): torch molecular model
        x        (torch.Tensor): node feature tensor [1, max_mol_size, nc_x]
        a        (torch.Tensor): adjacency tensor [1, max_mol_size, max_mol_size, nc_a]
        submol_size       (int): number of atoms of molecule made from 'x' and 'a'
        num_samples       (int): number of molecules to sample
        max_mol_size      (int): maximal size of molecule
        atom_list   (list[int]): list of feasible atom ids
    Returns: 
        mol, sml (tuple[list, list]): conditionaly generated molecules\smiles
    """
    mx, ma = create_mask(max_mol_size, submol_size)

    with torch.no_grad():
        xc, ac = model.sample_conditional(x, a, mx, ma, num_samples)
    mol_sample = gs2mols(xc.squeeze(1), ac.squeeze(1), atom_list)
    sml_sample = [Chem.MolToSmiles(mol, kekuleSmiles=True) for mol in mol_sample]

    return mol_sample, sml_sample

def create_observed_mol(smile, max_mol_size, atom_list, device='cuda'):
    """Create an submolecule to condition on.
    Parameters:
        smile             (str): smile of submolecule to condition on
        atom_list   (list[int]): list of feasible atom ids
        device       (str): target device of new mask
    Returns: 
        xx (torch.Tensor): node feature tensor [1, max_mol_size, nc_x]
        aa (torch.Tensor): adjacency tensor    [1, max_mol_size, max_mol_size, nc_a]
        submol_size (int): number of atoms of molecule made from 'smile'
    """
    mol = Chem.MolFromSmiles(smile)
    Chem.Kekulize(mol)
    xx, aa = mol2g(mol, max_mol_size, atom_list)
    xx, aa = xx.unsqueeze(0).to(device).long(), aa.unsqueeze(0).to(device).long()
    submol_size = mol.GetNumAtoms()
    return xx, aa, submol_size

if __name__ == "__main__":
    dataset = 'qm9'
    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']
    max_types = MOLECULAR_DATASETS[dataset]['max_types']

    # trained model path
    model_path = "/home/rektomar/projects/MolSPN/results/training/model_checkpoint/qm9/molspn_vaef_sort/dataset=qm9_model=molspn_vaef_sort_nx=9_nx_x=5_nx_a=4_nz=9_nz_x=5_nz_a=4_nz_y=0_h_x=2048_h_a=1024_h_y=512_l_x=8_l_a=8_l_y=4_l_b=4_ftype=none_device=cuda_lr=0.0001_weight_decay=0.0_num_epochs=100_batch_size=256_seed=0.pt"
    model = torch.load(model_path, weights_only=False)
    torch.manual_seed(2)

    num_samples = 1000
    num_to_show = 8  # assuming at least num_to_show of num_samples are valid
    # nice utility for molecule drawings https://www.rcsb.org/chemical-sketch
    patt_smls = ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'C1=CC=CC=C1', 'C1CN1C', 'N1C=CC=C1', 'COC']
    cond_smls = []

    for patt in patt_smls:
        xx, aa, submol_size = create_observed_mol(patt, max_atoms, atom_list)
        mols, smls = conditional_sample(model, xx, aa, submol_size, num_samples, max_atoms, atom_list)
        valid_smls = [sml for mol, sml in zip(mols, smls) if isvalid(mol)]
        valid_mols = [mol for mol in mols if isvalid(mol)]

        # small molecule filtering
        small_smls = [sml for mol, sml in zip(valid_mols[10:], valid_smls[10:]) if len(mol.GetAtoms())-submol_size<submol_size-1]
        final_smls = valid_smls[:8] if len(small_smls) < 3 else valid_smls[:6] + [small_smls[0], small_smls[2]]
        print(len(final_smls))
        print(final_smls)

        cond_smls.append(final_smls)
    
    grid_conditional(cond_smls, patt_smls, useSVG=False)
    grid_unconditional(model, 8, 8, max_atoms, atom_list, useSVG=False)

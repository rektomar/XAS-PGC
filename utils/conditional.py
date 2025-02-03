import torch

from rdkit import Chem, rdBase
rdBase.DisableLog("rdApp.error")

from utils.molecular import mol2g, gs2mols, isvalid
from utils.evaluate import evaluate_molecules, print_metrics
from utils.datasets import load_dataset

import numpy as np


def create_mask(batch_size, max_mol_size, submol_size, device='cuda'):
    """Create marginalization masks 'mx' and 'ma' for a single molecule.
       E.g., mx[j] = 1 means j-th node is observed,
             mx[j] = 0 means j-th node is marginalized out,
    Parameters:
        max_mol_size (int): maximal number of atoms in molecule
        submol_size  (int): number of atoms of masked submolecule 
                            submol_size < max_mol_size
        device       (str): target device of new mask
    Returns:
        mx (torch.Tensor): node feature mask     [batch_size, max_mol_size]
        ma (torch.Tensor): adjacency tensor mask [batch_size, max_mol_size, max_mol_size]
    """
    assert submol_size < max_mol_size
    mx = torch.zeros(batch_size, max_mol_size,               dtype=torch.bool, device=device)
    ma = torch.zeros(batch_size, max_mol_size, max_mol_size, dtype=torch.bool, device=device)
    mx[:, :submol_size              ] = True
    ma[:, :submol_size, :submol_size] = True
    return mx, ma

def sample_conditional(model, x, a, submol_size, num_samples, max_mol_size, atom_list):
    """Conditionaly generate a molecule given some other (smaller) molecule.
    Parameters:
        model (torch.nn.Module): torch molecular model
        x        (torch.Tensor): node feature tensor [num_samples, max_mol_size, nc_x]
        a        (torch.Tensor): adjacency tensor [num_samples, max_mol_size, max_mol_size, nc_a]
        submol_size       (int): number of atoms of molecule made from 'x' and 'a'
        num_samples       (int): number of molecules to sample
        max_mol_size      (int): maximal size of molecule
        atom_list   (list[int]): list of feasible atom ids
    Returns: 
        mol, sml (tuple[list, list]): conditionaly generated molecules\smiles
    """
    mx, ma = create_mask(len(x), max_mol_size, submol_size)

    x[~mx] = 0
    a[~ma] = -1

    xc, ac = model.sample(cond_x=x, cond_a=a)
    mol_sample = gs2mols(xc, ac, atom_list)
    sml_sample = [Chem.MolToSmiles(mol, kekuleSmiles=True) for mol in mol_sample]

    return xc, ac, mol_sample, sml_sample

def create_observed_mol(smile, max_mol_size, atom_list, device='cuda'):
    """Create a submolecule to condition on.
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
    xx, aa = xx.unsqueeze(0).float().to(device), aa.unsqueeze(0).float().to(device)
    submol_size = mol.GetNumAtoms()
    return xx, aa, submol_size

def filter_molecules(smls, patt_sml):
    patt = Chem.MolFromSmarts(patt_sml)
    f = lambda sml: Chem.MolFromSmiles(sml).HasSubstructMatch(patt)
    fsmls = list(filter(f, smls))
    return fsmls

def evaluate_conditional(model, patt_sml, dataset_name, max_atoms, atom_list, num_samples, batch_size=256, seed=0, order='canonical'):
    loaders = load_dataset(dataset_name, batch_size, split=[0.8, 0.1, 0.1], order=order, seed=seed)

    xx, aa, submol_size = create_observed_mol(patt_sml, max_atoms, atom_list)
    xx = xx.expand(num_samples, -1).clone()
    aa = aa.expand(num_samples, -1, -1).clone()
    torch.manual_seed(seed)
    xc, ac, _, _ = sample_conditional(model, xx, aa, submol_size, num_samples, max_atoms, atom_list)

    fsmls_trn = filter_molecules(loaders['smiles_trn'], patt_sml)
    fsmls_val = filter_molecules(loaders['smiles_val'], patt_sml)
    fsmls_tst = filter_molecules(loaders['smiles_tst'], patt_sml)

    occs = {}
    occs['occ_trn'] = len(fsmls_trn)/len(loaders['smiles_trn']) if len(loaders['smiles_trn']) != 0 else 0
    occs['occ_val'] = len(fsmls_val)/len(loaders['smiles_val']) if len(loaders['smiles_val']) != 0 else 0
    occs['occ_tst'] = len(fsmls_tst)/len(loaders['smiles_tst']) if len(loaders['smiles_tst']) != 0 else 0

    loaders['smiles_trn'] = fsmls_trn
    loaders['smiles_val'] = fsmls_val
    loaders['smiles_tst'] = fsmls_tst

    # avg size increase (atoms, edges)

    mols_sampled = gs2mols(xc, ac, atom_list)

    patt_natoms = Chem.MolFromSmiles(patt_sml).GetNumAtoms()
    patt_nbonds = Chem.MolFromSmiles(patt_sml).GetNumBonds()

    natoms_inc = [m.GetNumAtoms()-patt_natoms for m in mols_sampled]
    nbonds_inc = [m.GetNumBonds()-patt_nbonds for m in mols_sampled]

    stats = {'nat_inc': np.mean(natoms_inc), 'nbo_inc': np.mean(nbonds_inc)}

    metrics = evaluate_molecules(xc, ac, loaders, atom_list, evaluate_trn=False,
                                                             evaluate_val=True,
                                                             evaluate_tst=True,
                                                             metrics_only=True)
    metrics = metrics | occs | stats

    return metrics

def create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, max_atoms, atom_list):
    assert num_to_show < num_to_sample
    cond_smls = []

    for patt in patt_smls:
        xx, aa, submol_size = create_observed_mol(patt, max_atoms, atom_list)
        xx = xx.expand(num_to_sample, -1).clone()
        aa = aa.expand(num_to_sample, -1, -1).clone()
        _, _, mols, smls = sample_conditional(model, xx, aa, submol_size, num_to_sample, max_atoms, atom_list)
        filtered = [(mol, sml) for (mol, sml) in zip(mols, smls) if isvalid(mol)]
        valid_mols, valid_smls = zip(*filtered)
        valid_mols, valid_smls = list(valid_mols), list(valid_smls)

        # TODO: num_to_show > num_valid case
        final_smls = valid_smls[:num_to_show]
        print(f"Pattern {patt}: {final_smls}")

        cond_smls.append(final_smls)
    return cond_smls

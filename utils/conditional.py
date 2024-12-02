import torch

from rdkit import Chem, rdBase
rdBase.DisableLog("rdApp.error")

from utils.molecular import mol2g, gs2mols, isvalid
from utils.evaluate import evaluate_molecules, print_metrics
from utils.datasets import load_dataset


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

def sample_conditional(model, x, a, submol_size, num_samples, max_mol_size, atom_list):
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
        xc, ac = xc.squeeze(1), ac.squeeze(1)
    mol_sample = gs2mols(xc, ac, atom_list)
    sml_sample = [Chem.MolToSmiles(mol, kekuleSmiles=True) for mol in mol_sample]

    return xc, ac, mol_sample, sml_sample

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
    xx, aa = xx.unsqueeze(0).float().to(device), aa.unsqueeze(0).float().to(device)
    submol_size = mol.GetNumAtoms()
    return xx, aa, submol_size

def filter_molecules(smls, patt_sml):
    patt = Chem.MolFromSmarts(patt_sml)
    f = lambda sml: Chem.MolFromSmiles(sml).HasSubstructMatch(patt)
    fsmls = list(filter(f, smls))
    return fsmls

def evaluate_conditional(model, patt_sml, dataset_name, max_atoms, atom_list, num_samples, batch_size=256, order='canonical'):
    print(f'Evaluating "{patt_sml}" pattern on {dataset_name} dataset')
    loader_trn, _ = load_dataset(dataset_name, batch_size, split=[0.8, 0.2], order=order)
    train_smls = [x['s'] for x in loader_trn.dataset]

    xx, aa, submol_size = create_observed_mol(patt_sml, max_atoms, atom_list)
    xc, ac, _, _ = sample_conditional(model, xx, aa, submol_size, num_samples, max_atoms, atom_list)

    train_fsmls = filter_molecules(train_smls, patt_sml)
    occ_pct = 100*len(train_fsmls)/len(train_smls)
    metrics = evaluate_molecules(xc, ac, train_fsmls, atom_list, metrics_only=True)
    print(f'\tPattern occurence in training dataset: {occ_pct:.2f}%')
    print('\t', end='')
    print_metrics(metrics)

def create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, max_atoms, atom_list):
    assert num_to_show < num_to_sample
    cond_smls = []

    for patt in patt_smls:
        xx, aa, submol_size = create_observed_mol(patt, max_atoms, atom_list)
        _, _, mols, smls = sample_conditional(model, xx, aa, submol_size, num_to_sample, max_atoms, atom_list)
        filtered = [(mol, sml) for (mol, sml) in zip(mols, smls) if isvalid(mol)]
        valid_mols, valid_smls = zip(*filtered)
        valid_mols, valid_smls = list(valid_mols), list(valid_smls)

        # small molecule filtering
        small_smls = [sml for mol, sml in zip(valid_mols[num_to_show:], valid_smls[num_to_show:]) if len(mol.GetAtoms())-submol_size<submol_size-1]
        # print(valid_smls, small_smls)
        final_smls = valid_smls[:num_to_show] if len(small_smls) < 2 else valid_smls[:num_to_show-2] + small_smls[:2]
        print(f"Pattern {patt}: {final_smls}")

        cond_smls.append(final_smls)
    return cond_smls

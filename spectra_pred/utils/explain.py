import torch

from utils.metrics import rse
from utils.datasets import MIN_E, MAX_E, N_GRID

from rdkit import Chem
from utils.molecular import valency, BOND_DECODER, VALENCY_LIST, EMPTY_NODE_CAT, EMPTY_EDGE_CAT

import itertools

# an utility to covert a graph into a molecule with atoms and bonds to be highlighted
def g2mol_e(x, a, atom_list, node_mask):
    mol = Chem.RWMol()
    atoms_exist = ~(a == 0).all(dim=0) & (x != 0)

    atoms = x[atoms_exist]

    ar = torch.arange(len(atoms))
    hit_mask = node_mask[atoms_exist].bool()
    hit_atoms = ar[hit_mask].tolist()
    hit_bonds = []

    for atom in atoms:
        mol.AddAtom(Chem.Atom(atom_list[atom]))

    bonds = a[atoms_exist, :][:, atoms_exist]
    for start, end in zip(*torch.nonzero(bonds != 0, as_tuple=True)):
        if start > end:
            mol.AddBond(int(start), int(end), BOND_DECODER[bonds[start, end].item()])
            if int(start) in hit_atoms and int(end) in hit_atoms:
                bond_id = mol.GetBondBetweenAtoms(int(start), int(end)).GetIdx()
                hit_bonds.append(bond_id)
            flag, valence = valency(mol)
            if flag:
                continue
            else:
                assert len(valence) == 2
                i = valence[0]
                v = valence[1]
                a = mol.GetAtomWithIdx(i).GetAtomicNum()
                if a in (7, 8, 16) and (v - VALENCY_LIST[a]) == 1:
                    mol.GetAtomWithIdx(i).SetFormalCharge(1)
    return mol, hit_atoms, hit_bonds


def choose_range(lb_e, ub_e):
    energies = torch.linspace(MIN_E, MAX_E, N_GRID)
    return (energies >= lb_e) & (energies <= ub_e)

def mae(y_target, y_prediction):
    return (y_target-y_prediction).abs().mean()

def mask_graph(x, a, node_mask):
    node_mask = node_mask.bool()
    x_, a_ = x.clone(), a.clone()
    x_[..., ~node_mask] = EMPTY_NODE_CAT
    a_[..., ~node_mask, :] = EMPTY_EDGE_CAT
    a_[..., :, ~node_mask] = EMPTY_EDGE_CAT
    return x_, a_

def predict(model, x, a, out_mask, node_mask=None):
    if node_mask is None:
        x_masked, a_masked = x, a
    else:
        x_masked, a_masked = mask_graph(x, a, node_mask)
    return model.predict(x_masked, a_masked)[..., out_mask]

def is_valid(model, x, a, out_mask, node_mask, y_initial, threshold=0.1):
    y_pred = predict(model, x, a, out_mask, node_mask)
    return mae(y_initial, y_pred) < threshold

def prepare_input(x, a):
    node_mask = torch.ones(x.shape[-1], dtype=torch.bool)
    node_mask[x == EMPTY_NODE_CAT] = 0
    n_atom = torch.sum(node_mask)
    x = x.unsqueeze(0)
    a = a.unsqueeze(0)
    return x, a, node_mask, n_atom

def forward_search(model, x, a, out_mask=None, threshold=0.5):
    if out_mask is None:
        out_mask = choose_range(MIN_E, MAX_E)
    x, a, node_mask, n_atom = prepare_input(x, a)
    y_initial = predict(model, x, a, out_mask)
    stack = [(node_mask.clone(), 0)]

    result = []

    while len(stack) > 0:
        current_node_mask, current_depth = stack.pop()
        
        if current_depth >= n_atom:
            y_pred = predict(model, x, a, out_mask, current_node_mask)
            result.append([current_node_mask.clone().int().tolist(), current_node_mask.sum(), mae(y_initial, y_pred).item()])
            
            print(f"FOUND SOLUTION! size: {current_node_mask.sum()}, mae: {mae(y_initial, y_pred).item():.3f}, mask: {current_node_mask.int().tolist()}")
            continue

        stack.append((current_node_mask.clone(), current_depth+1))

        current_node_mask[current_depth] = False
        if is_valid(model, x, a, out_mask, current_node_mask, y_initial, threshold):
            stack.append((current_node_mask.clone(), current_depth+1))
    
    return result

def bruteforce_search(model, x, a, out_mask=None):

    if out_mask is None:
        out_mask = choose_range(MIN_E, MAX_E)
    x, a, node_mask, n_atom = prepare_input(x, a)
    max_atom = x.shape[-1]
    y_initial = predict(model, x, a, out_mask)

    result = {i: [node_mask, i, float('inf')] for i in range(0, 10)}


    for node_mask in itertools.product([False, True], repeat=max_atom):
        current_node_mask = torch.tensor(node_mask, dtype=bool)
        n = torch.sum(current_node_mask).item()
        y_pred = predict(model, x, a, out_mask, current_node_mask)
        current_mae = mae(y_initial, y_pred).item()
        if current_mae < result[n][2]:
            result[n] = [current_node_mask.clone().int().tolist(), n, current_mae]
    
    return list(result.values())


if __name__ == '__main__':
    
    path_model = '/home/rektomar/projects/XAS-PGC/spectra_pred/results/trn/ckpt/qm9xas/ffnn_zero_sort/dataset=qm9xas_order=canonical_model=ffnn_zero_sort_nd_n=9_nk_n=5_nk_e=4_nd_y=100_nl=8_device=cuda_lr=0.01_betas=[0.9, 0.999]_weight_decay=0.0_transform=normal_num_epochs=100_batch_size=256_seed=0.pt'
    model = torch.load(path_model, weights_only=False)
    
    from utils.datasets import MOLECULAR_DATASETS, load_dataset
    data_info = MOLECULAR_DATASETS['qm9']

    loader = load_dataset('qm9xas_canonical', 100, [0.8, 0.1, 0.1])['loader_tst'] 
    batch = next(iter(loader))
    id = 1
    x, a, spec, smile = batch['x'][id], batch['a'][id], batch['spec'][id], batch['s'][id]
    print(smile)

    e_mask = choose_range(280, 290)
    # result = forward_search(model.cpu(), x, a, e_mask)
    result = bruteforce_search(model.cpu(), x, a, e_mask)
    print(result)

    m = torch.tensor(result[3][0])

    mol, hit_atoms, hit_bonds = g2mol_e(x, a, data_info['atom_list'], m)

    from rdkit.Chem import Draw

    img = Draw.MolToImage(
        mol,
        highlightAtoms=hit_atoms,
        highlightBonds=hit_bonds,
    )

    img.save('mol.png')


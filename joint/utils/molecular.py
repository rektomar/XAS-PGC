import re
import torch

from rdkit import Chem


# VALENCY_LIST has to change for different datasets.
VALENCY_LIST = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}

BOND_ENCODER = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
BOND_DECODER = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def unpad(x, a):
    atoms_exist = (a.sum(dim=0) > 0) & (x != 0)
    atoms = x[atoms_exist]
    bonds = a[atoms_exist, :][:, atoms_exist]
    return atoms, bonds

def mol2x(mol, max_atom, atom_list):
    atom_tensor = torch.zeros(max_atom, dtype=torch.int8)
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        atom_tensor[atom_idx] = atom_list.index(atom.GetAtomicNum())
    return atom_tensor

def mol2a(mol, max_atom):
    bond_tensor = torch.zeros(max_atom, max_atom, dtype=torch.int8)
    for bond in mol.GetBonds():
        c = BOND_ENCODER[bond.GetBondType()]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_tensor[i, j] = c
        bond_tensor[j, i] = c
    return bond_tensor

def mol2g(mol, max_atom, atom_list):
    x = mol2x(mol, max_atom, atom_list)
    a = mol2a(mol, max_atom)
    return x, a

def g2mol(x, a, atom_list):
    mol = Chem.RWMol()

    atoms, bonds = unpad(x, a)

    for atom in atoms:
        mol.AddAtom(Chem.Atom(atom_list[atom]))

    for start, end in zip(*torch.nonzero(bonds > 0, as_tuple=True)):
        if start > end:
            mol.AddBond(int(start), int(end), BOND_DECODER[bonds[start, end].item()])
            flag, valence = valency(mol)
            if flag:
                continue
            else:
                assert len(valence) == 2
                i = valence[0]
                v = valence[1]
                n = mol.GetAtomWithIdx(i).GetAtomicNum()
                if n in (7, 8, 16) and (v - VALENCY_LIST[n]) == 1:
                    mol.GetAtomWithIdx(i).SetFormalCharge(1)
    return mol

def mols2gs(mols, max_atom, atom_list):
    x = torch.stack([mol2x(mol, max_atom, atom_list) for mol in mols])
    a = torch.stack([mol2a(mol, max_atom) for mol in mols])
    return x, a

def gs2mols(x, a, atom_list):
    return [g2mol(x, a, atom_list) for x, a in zip(x, a)]

def mols2smls(mols, canonical=True):
    return [Chem.MolToSmiles(mol, canonical=canonical) for mol in mols]

def valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as error:
        valence = list(map(int, re.findall(r'\d+', str(error))))
        return False, valence

def correct(mol):
    while True:
        flag, atomid_valence = valency(mol)
        if flag:
            break
        else:
            assert len (atomid_valence) == 2
            queue = []
            for b in mol.GetAtomWithIdx(atomid_valence[0]).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                bond_index = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if bond_index > 0:
                    mol.AddBond(start, end, BOND_DECODER[bond_index])

    return mol

def correct_mols(x, a, atom_list):
    return [correct(mol) for mol in gs2mols(x, a, atom_list)]

def get_valid(sml):
    mol = Chem.MolFromSmiles(sml)
    if mol is not None and '.' not in sml:
        Chem.Kekulize(mol)
        return mol, sml
    else:
        return None

def get_vmols(smls):
    vmols = []
    vsmls = []
    for s in smls:
        v = get_valid(s)
        if v is not None:
            vmols.append(v[0])
            vsmls.append(v[1])

    return vmols, vsmls

def isvalid(mol, canonical=True):
    sml = Chem.MolToSmiles(mol, canonical=canonical)
    if Chem.MolFromSmiles(sml) is not None and mol is not None and '.' not in sml:
        return True
    else:
        return False


if __name__ == '__main__':
    # 10 samples from the QM9 dataset
    max_atom = 9
    atom_list = [0, 6, 7, 8, 9]
    smiles = [
            'CCC1(C)CN1C(C)=O',
            'O=CC1=COC(=O)N=C1',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]

    for sa in smiles:
        mol = Chem.MolFromSmiles(sa)
        Chem.Kekulize(mol)
        x, a = mol2g(mol, max_atom, atom_list)
        mol = g2mol(x, a, atom_list)
        sb = Chem.MolToSmiles(mol, kekuleSmiles=True)
        print(sa)
        print(sb)

import pickle
import networkx as nx
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Contrib.SA_Score.sascorer import calculateScore


def calculate_plogP(mol: Chem.rdchem.Mol):
    logP = Chem.Descriptors.MolLogP(mol)
    SA = calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6

    return logP - SA - cycle_length

def calculate_props(mol):
    Chem.Kekulize(mol)
    # molecule needs to be sanitized to add the necessary information to mol (such as 'RingInfo')
    Chem.SanitizeMol(mol)
    return {
        'SA'    : calculateScore(mol),        # Synthetic Accessibility
        'logP'  : Descriptors.MolLogP(mol),   # logP score (Hydrophobicity)
        'MW'    : Descriptors.MolWt(mol),     # Molecular Weight
        'QED'   : Chem.QED.qed(mol),          # Quantitative Estimate of Drug-likeness
        'plogP' : calculate_plogP(mol),       # Penalized logP score
        'TPSA'  : Descriptors.TPSA(mol),      # Topological Polar Surface Area
        'numRB' : Descriptors.NumRotatableBonds(mol),
        'BCT'   : Descriptors.BertzCT(mol),
        'numHBA': rdMolDescriptors.CalcNumHBA(mol),  # H-Bond acceptors
        'numHBD': rdMolDescriptors.CalcNumHBD(mol),  # H-Bond donors
        'numAlR': rdMolDescriptors.CalcNumAliphaticRings(mol), 
        'numArR': rdMolDescriptors.CalcNumAromaticRings(mol)  
         
        }

def calculate_props_df(mols):
    props = []
    for mol in mols:
        try:
            props.append(calculate_props(mol))
        except ZeroDivisionError:  # Zero division problem with SA calculation for really simple molecules
            pass
    return pd.DataFrame(props)

if __name__ == '__main__':
    mol = Chem.MolFromSmiles('CC(C)C1=CC(=CC=C1C2=NC3=CC=CC=C3N2C4=CC=CC=C4)C5=CC=C(C=C5)OC')
    print(calculate_props(mol))




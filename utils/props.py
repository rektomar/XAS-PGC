import pickle
import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score.sascorer import calculateScore

# from sklearn import svm  # for DRD2 scorer

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

### the commented section uses external SVM classsifier to to get DRD2 score
### and the classifier needs a deprecated version of sklearn
# drd2_clf_model = None
# def load_model_drd2():
#     global drd2_clf_model
#     with open('data/clf_py36.pkl', "rb") as f:
#         drd2_clf_model = pickle.load(f)

# def calculate_DRD2(mol):
#     if drd2_clf_model is None:
#         load_model_drd2()

#     mol = Chem.MolFromSmiles(smile)
#     fp = fingerprints_from_mol(mol)
#     score = drd2_clf_model.predict_proba(fp)[:, 1]
#     return float(score)

# def fingerprints_from_mol(mol):
#     fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
#     size = 2048
#     nfp = np.zeros((1, size), np.int32)
#     for idx,v in fp.GetNonzeroElements().items():
#         nidx = idx%size
#         nfp[0, nidx] += int(v)
#     return nfp

def calculate_props(mol):
    # print(Chem.MolToSmiles(mol))
    Chem.Kekulize(mol)
    # molecule needs to be sanitized to add the necessary information to mol (such as 'RingInfo')
    Chem.SanitizeMol(mol)
    return {
        'SA'   : calculateScore(mol),             # Synthetic Accessibility
        'logP' : Chem.Descriptors.MolLogP(mol),   # logP score
        'MW'   : Chem.Descriptors.MolWt(mol),     # Molecular Weight
        'QED'  : Chem.QED.qed(mol),               # Quantitative Estimate of Drug-likeness
        'plogP': calculate_plogP(mol),            # Penalized logP score
        # 'DRD2' : calculate_DRD2(mol)
        }

if __name__ == '__main__':
    mol = Chem.MolFromSmiles('C1CCCCC1')
    print(calculate_props(mol))




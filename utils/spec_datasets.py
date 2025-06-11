import os
import json
import urllib.request
import torch
import urllib
import pandas
import numpy as np

from rdkit import Chem
from tqdm import tqdm
from rdkit import RDLogger

from utils.molecular import mol2g, g2mol

from utils.broadening import batch_broadening


BASE_DIR = 'results/'

MOLECULAR_DATASETS = {
    'qm9': {
        'dataset': 'qm9',
        'max_atoms': 9,
        'max_types': 5,
        'atom_list': [0, 6, 7, 8, 9],
        'valency_dict': {6:4, 7:3, 8:2, 9:1}
    },
}


def process_spectra(data_spectra):
    spectra_stk = torch.tensor(data_spectra['spec_stk'])

    # Seperating the spectra that has different shape than others
    shape_lst = [312, 264, 216, 360, 312, 352, 440]
    spectra1_stk = torch.split(spectra_stk[:2256,], shape_lst)
    spectra2_stk = torch.stack(torch.split(spectra_stk[2256:,], 500))

    energies = torch.linspace(270, 300, 100)
    sigma = torch.tensor(0.8)

    broadened_spectra1 = batch_broadening(spectra1_stk, sigma, energies)
    broadened_spectra2 = batch_broadening(spectra2_stk, sigma, energies)

    return broadened_spectra1 + broadened_spectra2 

    
def download_qm9xas(dir='data/'):
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    spec_file = f'{dir}qm9_Cedge_xas_56k.npz'
    url_spec = 'https://zenodo.org/records/8276902/files/qm9_Cedge_xas_56k.npz'
    print('Downloading QM9 spectra.')
    urllib.request.urlretrieve(url_spec, spec_file)
    print('Downloaded QM9 spectra.')

    # url_mols = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
    # urllib.request.urlretrieve(url_mols, file)

    # with zipfile.ZipFile(file, 'r') as zip_ref:
    #    zip_ref.extractall(dir)

    mol_file = f'{dir}qm9.csv'
    url_csv = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
    print('Downloading QM9 molecules.')
    urllib.request.urlretrieve(url_csv, mol_file)
    print('Downloaded QM9 molecules.')

    preprocess('qm9xas', mol_file, spec_file, MOLECULAR_DATASETS['qm9']['max_atoms'], MOLECULAR_DATASETS['qm9']['atom_list'])



def preprocess(dataset, mol_path, spec_path, max_atom, atom_list, order='canonical'):

    data_spectra = np.load(spec_path)
    spectra = process_spectra(data_spectra)

    # suppl = Chem.SDMolSupplier('data/gdb9.sdf', removeHs=False, sanitize=True, strictParsing=False)
    input_df = pandas.read_csv(mol_path, sep=',', dtype='str')
    smls_list = list(input_df['smiles'])

    data_list = []

    for sml, spec in tqdm(zip(smls_list, spectra)):
        mol = Chem.MolFromSmiles(sml)
        Chem.Kekulize(mol)
        s = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
        n = mol.GetNumAtoms()
        x, a = mol2g(mol, max_atom, atom_list)
        data_list.append({'x': x, 'a': a, 'n': n, 's': s, 'spec': spec})

    torch.save(data_list, f'data/{dataset}_{order}.pt')


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def fit_transform(x):
    spectra = torch.stack([x_i['spec'] for x_i in x])
    return torch.mean(spectra, dim=0), torch.std(spectra, dim=0)

def transform(x, mu, sigma):
    for x_i in x:
        x_i['spec'] = (x_i['spec']-mu)/sigma  
    return x

def load_dataset(name, batch_size, split, seed=0, dir='data/', order='canonical'):
    x = DictDataset(torch.load(f'{dir}{name}_{order}.pt', weights_only=True))

    torch.manual_seed(seed)
    x_trn, x_val, x_tst = torch.utils.data.random_split(x, split)

    mu, sigma = fit_transform(x_trn)

    x_trn = transform(x_trn, mu, sigma)
    x_val = transform(x_val, mu, sigma)
    x_tst = transform(x_tst, mu, sigma)

    loader_trn = torch.utils.data.DataLoader(x_trn, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(x_val, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
    loader_tst = torch.utils.data.DataLoader(x_tst, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)

    smiles_trn = [x['s'] for x in loader_trn.dataset]
    smiles_val = [x['s'] for x in loader_val.dataset]
    smiles_tst = [x['s'] for x in loader_tst.dataset]

    return {
        'loader_trn': loader_trn,
        'loader_val': loader_val,
        'loader_tst': loader_tst,
        'smiles_trn': smiles_trn,
        'smiles_val': smiles_val,
        'smiles_tst': smiles_tst,
        'transform': (mu, sigma)
    }


if __name__ == '__main__':
    download_qm9xas()
    loaders = load_dataset('qm9xas', 100, split=[0.8, 0.1, 0.1])

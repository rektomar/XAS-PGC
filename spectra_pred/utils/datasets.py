import os
import json
import urllib.request
import torch
import urllib
import pandas
import numpy as np

import torch.nn as nn

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

KERNEL_WIDTH = 0.8
MIN_E = 270
MAX_E = 300
N_GRID = 100

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
    

class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    def inverse(self, z):
        return z

def collect(dataset, key):
    return torch.stack([x[key] for x in dataset])

class Standardize(nn.Module):

    def __init__(self, x):
        super().__init__()
        self.fit(x)
    
    def fit(self, dataset):
        x = collect(dataset, 'spec')
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)

    def forward(self, x):
        return (x - self.mean)/self.std
    
    def inverse(self, z):
        return self.std * z + self.mean
    
class LogStandardize(nn.Module):
    
    def __init__(self, x, log_eps=1e-8):
        super().__init__()
        self.log_eps = log_eps
        self.fit(x)
    
    def fit(self, dataset):
        x = collect(dataset, 'spec')
        log_x = torch.log(x+self.log_eps)
        self.mean = torch.mean(log_x, dim=0)
        self.std = torch.std(log_x, dim=0)

    def forward(self, x):
        return (torch.log(x+self.log_eps) - self.mean)/self.std
    
    def inverse(self, z):
        return torch.maximum(torch.exp(self.std * z + self.mean) - self.log_eps, torch.zeros_like(z))


def transform_dataset(x, transform):
    for x_i in x:
        x_i['spec'] = transform(x_i['spec'])  
    return x


def load_dataset(name, batch_size, split, seed=0, dir='data/', standardize='normal'):
    x = DictDataset(torch.load(f'{dir}{name}.pt', weights_only=True))

    torch.manual_seed(seed)
    x_trn, x_val, x_tst = torch.utils.data.random_split(x, split)


    if standardize == 'none':
        transform = Identity()
    elif standardize == 'normal':
        transform = Standardize(x_trn)
    elif standardize == 'lognormal':
        transform = LogStandardize(x_trn)
    else:
        transform = None

    if transform is not None:
        transform_dataset(x_trn, transform)
        transform_dataset(x_val, transform)
        transform_dataset(x_tst, transform)


    loader_trn = torch.utils.data.DataLoader(x_trn, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(x_val, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
    loader_tst = torch.utils.data.DataLoader(x_tst, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)


    return {
        'loader_trn': loader_trn,
        'loader_val': loader_val,
        'loader_tst': loader_tst,
        'transform': transform
    }

if __name__ == '__main__':
    download_qm9xas()
    loaders = load_dataset('qm9xas_canonical', 100, split=[0.8, 0.1, 0.1])

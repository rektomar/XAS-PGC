import os
import json
import torch
import urllib
import pandas

from rdkit import Chem
from tqdm import tqdm
from rdkit import RDLogger

from utils.molecular import mol2g, g2mol
from utils.graphs import permute_graph, flatten, bandwidth, unflatten
from utils.evaluate import evaluate_molecules
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

MOLECULAR_DATASETS = {
    'qm9': {
        'dataset': 'qm9',
        'max_atoms': 9,
        'max_types': 5,
        'atom_list': [6, 7, 8, 9, 0]
    },
    'zinc250k': {
        'dataset': 'zinc250k',
        'max_atoms': 38,
        'max_types': 10,
        'atom_list': [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    }
}


def download_qm9(dir='data/', order='canonical'):
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}qm9'
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/qm9_property.csv'

    print('Downloading and preprocessing the QM9 dataset.')

    urllib.request.urlretrieve(url, f'{file}.csv')
    preprocess(file, 'smile', 'penalized_logp', MOLECULAR_DATASETS['qm9']['max_atoms'], MOLECULAR_DATASETS['qm9']['atom_list'], order)
    os.remove(f'{file}.csv')

    print('Done.')

def download_zinc250k(dir='data/', order='canonical'):
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}zinc250k'
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/zinc250k_property.csv'

    print('Downloading and preprocessing the Zinc250k dataset.')

    urllib.request.urlretrieve(url, f'{file}.csv')
    preprocess(file, 'smile', 'penalized_logp', MOLECULAR_DATASETS['zinc250k']['max_atoms'], MOLECULAR_DATASETS['zinc250k']['atom_list'], order)
    os.remove(f'{file}.csv')

    print('Done.')


def preprocess(path, smile_col, prop_name, max_atom, atom_list, order='canonical'):
    input_df = pandas.read_csv(f'{path}.csv', sep=',', dtype='str')
    smls_list = list(input_df[smile_col])
    prop_list = list(input_df[prop_name])
    data_list = []
    max_bandwidth = 0

    for smls, prop in tqdm(zip(smls_list, prop_list)):
        mol = Chem.MolFromSmiles(smls)
        Chem.Kekulize(mol)
        n = mol.GetNumAtoms()
        x, a = mol2g(mol, max_atom, atom_list)
        y = torch.tensor([float(prop)])

        if order == 'canonical':
            s = Chem.MolToSmiles(mol, kekuleSmiles=True)
            data_list.append({'x': x, 'a': a, 'n': n, 's': s, 'y': y})

        elif order == 'mc':
            s = Chem.MolToSmiles(mol, kekuleSmiles=True)
            p = reverse_cuthill_mckee(csr_matrix((a != 3).to(torch.int8)))
            x, a = permute_graph(x, a, p.copy())
            b = bandwidth((a != 3).to(torch.int8), n)
            a = flatten(a, b)
            if b > max_bandwidth:
                max_bandwidth = b
            data_list.append({'x': x, 'a': a, 'b': b, 'n': n, 's': s, 'y': y})

        elif order == 'rand':
            x, a = permute_graph(x, a, torch.cat((torch.randperm(n), torch.arange(n, max_atom))))
            s = Chem.MolToSmiles(g2mol(x, a, atom_list), kekuleSmiles=True, canonical=False)
            data_list.append({'x': x, 'a': a, 'n': n, 's': s, 'y': y})

        else:
            os.error('Unknown order')

    if order == 'mc':
        for data in data_list:
            data['a'] = torch.nn.functional.pad(data['a'], (0, max_bandwidth - data['b']), 'constant', 3)

    torch.save(data_list, f'{path}_{order}.pt')

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def load_dataset(name, batch_size, raw=False, seed=0, split=None, dir='data/', order='canonical'):
    x = DictDataset(torch.load(f'{dir}{name}_{order}.pt', weights_only=True))

    if split is None:
        with open(f'{dir}i_val_{name}.json') as f:
            i_val = json.load(f)
        i_trn = [t for t in range(len(x)) if t not in i_val]

        x_trn = torch.utils.data.Subset(x, i_trn)
        x_val = torch.utils.data.Subset(x, i_val)
    else:
        torch.manual_seed(seed)
        x_trn, x_val = torch.utils.data.random_split(x, split)

    if raw == True:
        return x_trn, x_val
    else:
        loader_trn = torch.utils.data.DataLoader(x_trn, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
        loader_val = torch.utils.data.DataLoader(x_val, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)

        return loader_trn, loader_val


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    torch.set_printoptions(threshold=10_000, linewidth=200)

    download = True
    dataset = 'zinc250k'
    order = 'rand'

    if download:
        if dataset == 'qm9':
            download_qm9(order=order)
        elif dataset == 'zinc250k':
            download_zinc250k(order=order)
        else:
            os.error('Unsupported dataset.')

    loader_trn, loader_val = load_dataset(dataset, 100, split=[0.99, 0.01], order=order)

    x = [e['x'] for e in loader_trn.dataset]
    a = [e['a'] for e in loader_trn.dataset]
    s = [e['s'] for e in loader_trn.dataset]

    x = torch.stack(x)
    a = torch.stack(a)

    if order == 'mc':
        r = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
        for i in range(x.shape[0]):
            r[i] = unflatten(a[i])
        a = r

    print(evaluate_molecules(x, a, s, MOLECULAR_DATASETS[dataset]['atom_list'], metrics_only=True, canonical=True))

    # loader_trn, loader_val = load_dataset(dataset, 100, split=[0.8, 0.2], canonical=False)

    # x = [e['x'] for e in loader_trn.dataset]
    # a = [e['a'] for e in loader_trn.dataset]
    # s = [e['s'] for e in loader_trn.dataset]

    # print(evaluate_molecules(x, a, s, MOLECULAR_DATASETS[dataset]['atom_list'], metrics_only=True, canonical=False))

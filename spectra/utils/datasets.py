import os
import urllib.request
import torch
import urllib
import numpy as np

import torch.nn as nn


from utils.broadening import batch_broadening

KERNEL_WIDTH = 0.8
MIN_E = 270
MAX_E = 300
N_GRID = 100

BASE_DIR = 'results/'


def process_spectra(data_spectra):
    spectra_stk = torch.tensor(data_spectra['spec_stk'])

    # Seperating the spectra that has different shape than others
    shape_lst = [312, 264, 216, 360, 312, 352, 440]
    spectra1_stk = torch.split(spectra_stk[:2256,], shape_lst)
    spectra2_stk = torch.stack(torch.split(spectra_stk[2256:,], 500))

    energies = torch.linspace(MIN_E, MAX_E, N_GRID)
    sigma = torch.tensor(KERNEL_WIDTH)

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

    # mol_file = f'{dir}qm9.csv'
    # url_csv = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
    # print('Downloading QM9 molecules.')
    # urllib.request.urlretrieve(url_csv, mol_file)
    # print('Downloaded QM9 molecules.')


    data_spectra = np.load(spec_file)
    spectra = process_spectra(data_spectra)

    torch.save(spectra, f'data/qm9xas.pt')

class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    def inverse(self, z):
        return z

class Standardize(nn.Module):

    def __init__(self, x):
        super().__init__()
        self.fit(x)
    
    def fit(self, x):
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
    
    def fit(self, x):
        log_x = torch.log(x+self.log_eps)
        self.mean = torch.mean(log_x, dim=0)
        self.std = torch.std(log_x, dim=0)

    def forward(self, x):
        return (torch.log(x+self.log_eps) - self.mean)/self.std
    
    def inverse(self, z):
        return torch.maximum(torch.exp(self.std * z + self.mean) - self.log_eps, torch.zeros_like(z))


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

def load_dataset(name, batch_size, split, seed=0, dir='data/', standardize='normal'):
    x = TensorDataset(torch.load(f'{dir}{name}.pt', weights_only=True))

    torch.manual_seed(seed)
    x_trn, x_val, x_tst = torch.utils.data.random_split(x, split)

    # print(x_trn.indices)
    data = torch.stack(x.data)
    x_train_tensor = data[x_trn.indices]
    x_val_tensor = data[x_val.indices]
    x_test_tensor = data[x_tst.indices]

    if standardize == 'none':
        transform = Identity()
    elif standardize == 'normal':
        transform = Standardize(x_train_tensor)
    elif standardize == 'lognormal':
        transform = LogStandardize(x_train_tensor)
    else:
        transform = None

    if transform is not None:
        x_train_tensor = transform(x_train_tensor)
        x_val_tensor = transform(x_val_tensor)
        x_test_tensor = transform(x_test_tensor)

    x_trn = TensorDataset(x_train_tensor)
    x_val = TensorDataset(x_val_tensor)
    x_tst = TensorDataset(x_test_tensor)

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
    # download_qm9xas()
    loaders = load_dataset('qm9xas', 256, split=[0.8, 0.1, 0.1])

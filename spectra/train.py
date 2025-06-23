import json
import torch

from rdkit import RDLogger
from utils.datasets import BASE_DIR, load_dataset
from utils.train import train #, evaluate
from utils.evaluate import count_parameters


from models import vae

MODELS = {
    **vae.MODELS
}

BASE_DIR_TRN = f'{BASE_DIR}trn/'

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'

    backends = [
        'vae'
    ]

    for backend in backends:
        with open(f'config/{dataset}/{backend}.json', 'r') as f:
            hyperpars = json.load(f)

        loaders = load_dataset('qm9xas', hyperpars['batch_size'], [0.8, 0.1, 0.1], standardize=hyperpars['standardize'])

        model = MODELS[hyperpars['model']](**hyperpars['model_hpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')

        train(model, loaders, hyperpars, BASE_DIR_TRN)
        # metrics = evaluate(loaders, hyperpars, BASE_DIR_TRN, compute_nll=True)

        # print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))

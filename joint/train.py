import json
import torch

from rdkit import RDLogger
from utils.spec_datasets import MOLECULAR_DATASETS, BASE_DIR, load_dataset
from utils.train import train, evaluate
from utils.evaluate import count_parameters

# from models import cm_var_spec

# MODELS = {
#     **cm_var_spec.MODELS
# }

from models import cm_spec

MODELS = {
    **cm_spec.MODELS
}

BASE_DIR_TRN = f'{BASE_DIR}trn/'

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'

    backends = [
        'pgc_ffn_spec'
    ]

    for backend in backends:
        with open(f'config/{dataset}/{backend}.json', 'r') as f:
            hyperpars = json.load(f)
        hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']

        loaders = load_dataset('qm9xas', hyperpars['batch_size'], [0.8, 0.1, 0.1], order=hyperpars['order'])

        model = MODELS[hyperpars['model']](**hyperpars['model_hpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')
        print(hyperpars['order'])

        train(model, loaders, hyperpars, BASE_DIR_TRN)
        metrics = evaluate(loaders, hyperpars, BASE_DIR_TRN, compute_nll=True)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))

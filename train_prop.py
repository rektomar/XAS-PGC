import json
import torch

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, BASE_DIR, load_dataset
from utils.train_prop import train_prop, evaluate
from utils.evaluate import count_parameters

# from models import molspn_zero
from models import molspn_marg_prop

MODELS = {
    # **molspn_zero.MODELS,
    **molspn_marg_prop.MODELS
}

BASE_DIR_TRN = f'{BASE_DIR}trn_prop/'

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # torch.set_printoptions(threshold=10_000, linewidth=200)
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'

    prop_name = 'logP'
    backends = [
        'marg_sort_btree'
        # 'zero_sort_vtree'
        # 'zero_sort_rtree'
        # 'zero_sort_ptree'
        # 'zero_sort_ctree'
    ]

    for backend in backends:
        with open(f'config/{dataset}/{backend}.json', 'r') as f:
            hyperpars = json.load(f)
        hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']

        hyperpars['model_hpars']['prop'] = prop_name

        loaders = load_dataset(dataset, hyperpars['batch_size'], [0.8, 0.1, 0.1], order=hyperpars['order'])

        model = MODELS[hyperpars['model']](loaders['loader_trn'], hyperpars['model_hpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')
        print(hyperpars['order'])

        train_prop(model, loaders, hyperpars, BASE_DIR_TRN)
        metrics = evaluate(loaders, hyperpars, BASE_DIR_TRN, compute_nll=True)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))

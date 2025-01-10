import json
import torch

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.train import train, evaluate
from utils.evaluate import count_parameters

from models import molspn_zero

MODELS = {
    **molspn_zero.MODELS
}


CHECKPOINT_DIR = 'results/training/model_checkpoint/'
EVALUATION_DIR = 'results/training/model_evaluation/'


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # torch.set_printoptions(threshold=10_000, linewidth=200)
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'

    backends = [
        # 'zero_sort_btree'
        # 'zero_sort_vtree'
        # 'zero_sort_rtree'
        # 'zero_sort_ptree'
        'zero_sort_ctree'
    ]

    for backend in backends:
        with open(f'config/{dataset}/{backend}.json', 'r') as f:
            hyperpars = json.load(f)
        hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']

        loaders = load_dataset(dataset, hyperpars['batch_size'], [0.8, 0.1, 0.1], order=hyperpars['order'])

        model = MODELS[hyperpars['model']](loaders['loader_trn'], hyperpars['model_hpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')
        print(hyperpars['order'])

        path = train(model, loaders, hyperpars, CHECKPOINT_DIR)
        model = torch.load(path, weights_only=False)
        metrics = evaluate(model, loaders, hyperpars, EVALUATION_DIR, compute_nll=True)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))

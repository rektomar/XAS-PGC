import json
import torch

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.train import train, evaluate, backend_hpars_prefix
from utils.evaluate import count_parameters

from models import molspn_zero
from models import molspn_marg
from models import molspn_none
from models import molspn_band
from models import molspn_hclt

MODELS = {
    **molspn_zero.MODELS,
    **molspn_marg.MODELS,
    **molspn_band.MODELS,
    **molspn_none.MODELS,
    **molspn_hclt.MODELS
}


CHECKPOINT_DIR = 'results/training/model_checkpoint/'
EVALUATION_DIR = 'results/training/model_evaluation/'


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # torch.set_printoptions(threshold=10_000, linewidth=200)
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'

    names = [
        'zero_sort',
        # 'molspn_marg_sort',
        # 'molspn_none_sort',
        # 'molspn_hclt_sort',
        # 'molspn_mclt_sort',
    ]

    for name in names:
        with open(f'config/{dataset}/{name}.json', 'r') as f:
            hyperpars = json.load(f)
        hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']

        loader_trn, loader_val = load_dataset(dataset, hyperpars['batch_size'], split=[0.8, 0.2], order=hyperpars['order'])
        smiles_trn = [x['s'] for x in loader_trn.dataset]

        model = MODELS[name](loader_trn, hyperpars['model_hpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')
        print(hyperpars['order'])

        path = train(model, loader_trn, loader_val, smiles_trn, hyperpars, CHECKPOINT_DIR)
        model = torch.load(path, weights_only=False)
        metrics = evaluate(model, loader_trn, loader_val, smiles_trn, hyperpars, EVALUATION_DIR, compute_nll=False)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))

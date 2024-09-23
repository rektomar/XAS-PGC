import json
import torch

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.train import train, evaluate
from utils.evaluate import count_parameters

from models import molspn_zero
from models import molspn_marg
from models import molspn_none
from models import molspn_norm
from models import molspn_back
from models import molspn_vaes
from models import moflow

MODELS = {
    **molspn_back.MODELS,
    **molspn_zero.MODELS,
    **molspn_marg.MODELS,
    **molspn_norm.MODELS,
    **molspn_none.MODELS,
    **molspn_vaes.MODELS,
    **moflow.MODELS
}


CHECKPOINT_DIR = 'results/training/model_checkpoint/'
EVALUATION_DIR = 'results/training/model_evaluation/'


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'
    names = [
        # 'molspn_ffnn_sort',
        # 'molspn_conv_sort',
        # 'molspn_flow_sort',
        # 'molspn_tran_sort',
        'molspn_zero_sort',
        # 'molspn_marg_sort',
        # 'molspn_none_sort',
        # 'molspn_norm_sort',
        # 'molspn_vaef_sort',
        # 'molspn_vaex_sort',
        # 'molspn_vaet_sort',
        # 'moflow_sort',
        # 'graphspn_zero_sort'
    ]

    for name in names:
        with open(f'config/{dataset}/{name}.json', 'r') as f:
            hyperpars = json.load(f)
        hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']
        hyperpars['max_atoms'] = MOLECULAR_DATASETS[dataset]['max_atoms']

        model = MODELS[name](**hyperpars['model_hyperpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')

        if 'sort' in name:
            canonical = True
        else:
            canonical = False

        loader_trn, loader_val = load_dataset(hyperpars['dataset'], hyperpars['batch_size'], split=[0.99, 0.01], canonical=canonical)
        smiles_trn = [x['s'] for x in loader_trn.dataset]

        path = train(model, loader_trn, loader_val, smiles_trn, hyperpars, CHECKPOINT_DIR)
        model = torch.load(path, weights_only=False)
        metrics = evaluate(model, loader_trn, loader_val, smiles_trn, hyperpars, EVALUATION_DIR, compute_nll=False, canonical=canonical)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))

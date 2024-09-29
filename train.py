import json
import torch

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.train import train, evaluate
from utils.evaluate import count_parameters

from models import molspn_zero
from models import molspn_marg
from models import molspn_none
from models import molspn_perm
from models import molspn_norm
from models import molspn_band
from models import molspn_back
from models import molspn_vaes
from models import moflow

MODELS = {
    **molspn_back.MODELS,
    **molspn_zero.MODELS,
    **molspn_marg.MODELS,
    **molspn_perm.MODELS,
    **molspn_band.MODELS,
    **molspn_norm.MODELS,
    **molspn_none.MODELS,
    **molspn_vaes.MODELS,
    **moflow.MODELS
}


CHECKPOINT_DIR = 'results/training/model_checkpoint/'
EVALUATION_DIR = 'results/training/model_evaluation/'


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # torch.set_printoptions(threshold=10_000, linewidth=200)
    RDLogger.DisableLog('rdApp.*')

    dataset = 'qm9'
    order = 'canonical'

    names = [
        # order = 'mc'
        # 'molspn_band_sort',

        # order = 'canonical'
        # 'molspn_ffnn_sort',
        # 'molspn_conv_sort',
        # 'molspn_flow_sort',
        # 'molspn_tran_sort',

        # 'molspn_zero_sort',
        # 'molspn_perm_sort',
        'molspn_marg_sort',
        # 'molspn_none_sort',
        # 'molspn_norm_sort',
        # 'molspn_vaef_sort',

        # need maintanence
        # 'molspn_vaet_sort',
        # 'moflow_sort',
        # 'graphspn_zero_sort'
    ]

    for name in names:
        with open(f'config/{dataset}/{name}.json', 'r') as f:
            hyperpars = json.load(f)
        hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']
        hyperpars['max_atoms'] = MOLECULAR_DATASETS[dataset]['max_atoms']

        loader_trn, loader_val = load_dataset(hyperpars['dataset'], hyperpars['batch_size'], split=[0.8, 0.2], order=order)
        smiles_trn = [x['s'] for x in loader_trn.dataset]

        if order == 'mc':
            hyperpars['model_hyperpars']['bw'] = loader_trn.dataset[0]['a'].size(-1)

        model = MODELS[name](**hyperpars['model_hyperpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')

        path = train(model, loader_trn, loader_val, smiles_trn, hyperpars, CHECKPOINT_DIR)
        model = torch.load(path, weights_only=False)
        metrics = evaluate(model, loader_trn, loader_val, smiles_trn, hyperpars, EVALUATION_DIR, compute_nll=False, canonical=True)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))

import json
import torch

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.train import train, evaluate
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
    order = 'canonical'
    # order = 'rand'
    # order = 'mc'

    names = [
        # order = 'mc'
        # 'molspn_band_sort',

        # order = 'canonical'
        'molspn_zero_sort',
        # 'molspn_marg_sort',
        # 'molspn_none_sort',
        # 'molspn_hclt_sort',
        # 'molspn_mclt_sort',
    ]

    for name in names:
        with open(f'config/{dataset}/{name}.json', 'r') as f:
            hyperpars = json.load(f)
        hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']
        hyperpars['max_atoms'] = MOLECULAR_DATASETS[dataset]['max_atoms']

        loader_trn, loader_val = load_dataset(hyperpars['dataset'], hyperpars['batch_size'], split=[0.8, 0.2], order=order)
        smiles_trn = [x['s'] for x in loader_trn.dataset]

        x = torch.stack([e['x'] for e in loader_trn.dataset])
        a = torch.stack([e['a'] for e in loader_trn.dataset])

        nd_n = MOLECULAR_DATASETS[dataset]['max_atoms']
        nd_e = nd_n * (nd_n - 1) // 2
        m = torch.tril(torch.ones(nd_n, nd_n, dtype=torch.bool), diagonal=-1)
        l = a[:, m].view(-1, nd_e)

        # if order == 'mc':
        #     hyperpars['model_hyperpars']['bw'] = loader_trn.dataset[0]['a'].size(-1)

        model = MODELS[name](**hyperpars['model_hyperpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')
        print(order)

        path = train(model, loader_trn, loader_val, smiles_trn, hyperpars, CHECKPOINT_DIR)
        model = torch.load(path, weights_only=False)
        metrics = evaluate(model, loader_trn, loader_val, smiles_trn, hyperpars, EVALUATION_DIR, compute_nll=False, canonical=True)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))

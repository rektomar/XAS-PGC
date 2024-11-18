import os
import time
import json
import torch
import subprocess
import gridsearch_hyperpars

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.train import train, evaluate, dict2str, flatten_dict, backend_hpars_prefix
from utils.evaluate import count_parameters

from models import molspn_zero

MODELS = {
    **molspn_zero.MODELS
    }


CHECKPOINT_DIR = 'results/gridsearch/model_checkpoint/'
EVALUATION_DIR = 'results/gridsearch/model_evaluation/'
OUTPUTLOGS_DIR = 'results/gridsearch/model_outputlogs/'


def unsupervised(dataset, name, par_buffer):
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    hyperpars = par_buffer[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']

    loader_trn, loader_val = load_dataset(dataset, hyperpars['batch_size'], split=[0.8, 0.2], order=hyperpars['order'])
    smiles_trn = [x['s'] for x in loader_trn.dataset]

    model = MODELS[name](loader_trn, hyperpars['model_hpars'])
    print(dataset)
    print(json.dumps(hyperpars, indent=4))
    print(model)
    print(f'The number of parameters is {count_parameters(model)}.')
    print(hyperpars['order'])

    path = train(model, loader_trn, loader_val, smiles_trn, hyperpars, CHECKPOINT_DIR, verbose=True)
    model = torch.load(path)
    metrics = evaluate(model, loader_trn, loader_val, smiles_trn, hyperpars, EVALUATION_DIR, compute_nll=False)

    print("\n".join(f'{key:<20}{value:>10.4f}' for key, value in metrics.items()))


def submit_job(dataset, model, par_buffer, device, max_sub):
    outputlogs_dir = OUTPUTLOGS_DIR + f'{dataset}/'
    par_buffer_str = str(par_buffer).replace("'", '"')
    cmd_python = "from gridsearch import unsupervised\n" + f'unsupervised("{dataset}", "{model}", {par_buffer_str})'
    cmd_sbatch = "source activate molspn\n" + f"python -c '{cmd_python}'"

    while True:
        run_squeue = subprocess.run(['squeue', f'--user={os.environ["USER"]}', '-h', '-r'], stdout=subprocess.PIPE)
        run_wcount = subprocess.run(['wc', '-l'], input=run_squeue.stdout, capture_output=True)
        num_queued = int(run_wcount.stdout)

        if len(par_buffer) <= max_sub - num_queued:
            if device == 'cuda':
                subprocess.run(['sbatch',
                                f'--job-name={model.replace("_","")}',
                                f'--output={outputlogs_dir}/{model}/%A_%a.out',
                                '--partition=amdgpufast',
                                '--ntasks=1',
                                '--mem-per-cpu=64000',
                                f'--gres=gpu:1',
                                f'--array=0-{len(par_buffer)-1}',
                                f'--wrap={cmd_sbatch}'])
            elif device == 'cpu':
                subprocess.run(['sbatch',
                                f'--job-name={model.replace("_","")}',
                                f'--output={outputlogs_dir}/{model}/%A_%a.out',
                                '--partition=amd',
                                '--ntasks=1',
                                '--ntasks-per-node=1',
                                '--cpus-per-task=1',
                                '--mem-per-cpu=64000',
                                f'--array=0-{len(par_buffer)-1}',
                                f'--wrap={cmd_sbatch}'])
            else:
                os.error('Unknown device.')

            break
        else:
            time.sleep(20)


if __name__ == "__main__":
    par_buffer = []
    all_models = [
        'zero_sort',
    ]
    gpu_models = MODELS.keys()

    for dataset in MOLECULAR_DATASETS.keys():
        print(dataset)
        for model in all_models:
            print(model)
            if model in gpu_models:
                device = 'cuda'
                max_sub = 20
                max_jobs_to_submit = 1
            else:
                device = 'cpu'
                max_sub = 500
                max_jobs_to_submit = 25

            for hyperpars in gridsearch_hyperpars.GRIDS[model](dataset):
                hyperpars['model_hpars']['device'] = device
                backend_hpars_prefix(hyperpars)

                path = EVALUATION_DIR + f'metrics/{dataset}/{model}/' + dict2str(flatten_dict(backend_hpars_prefix(hyperpars))) + '.csv'
                if not os.path.isfile(path):
                    par_buffer.append(hyperpars)

                if len(par_buffer) == max_jobs_to_submit:
                    submit_job(dataset, model, par_buffer, device, max_sub)
                    par_buffer = []

            if len(par_buffer) > 1:
                submit_job(dataset, model, par_buffer, device, max_sub)
                par_buffer = []

import os
import time
import json
import torch
import subprocess
import gridsearch_hyperpars

from rdkit import RDLogger
from utils.datasets import MOLECULAR_DATASETS, BASE_DIR, load_dataset
from utils.train import train, evaluate, dict2str, flatten_dict, backend_hpars_prefix
from utils.evaluate import count_parameters

from models import molspn_zero
from models import molspn_marg

# nohup python -m gridsearch > gridsearch.log &

MODELS = {
    **molspn_zero.MODELS,
    **molspn_marg.MODELS
    }

BASE_DIR_GS = f'{BASE_DIR}gs0/'


def unsupervised(dataset, name, par_buffer):
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    hyperpars = par_buffer[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    hyperpars['atom_list'] = MOLECULAR_DATASETS[dataset]['atom_list']

    loaders = load_dataset(dataset, hyperpars['batch_size'], [0.8, 0.1, 0.1], seed=hyperpars['seed'], order=hyperpars['order'])

    model = MODELS[name](loaders['loader_trn'], hyperpars['model_hpars'])
    print(dataset)
    print(json.dumps(hyperpars, indent=4))
    print(model)
    print(f'The number of parameters is {count_parameters(model)}.')
    print(hyperpars['order'])

    train(model, loaders, hyperpars, BASE_DIR_GS, verbose=True)
    metrics = evaluate(loaders, hyperpars, BASE_DIR_GS, compute_nll=True, verbose=True)

    print("\n".join(f'{key:<20}{value:>10.4f}' for key, value in metrics.items()))


def submit_job(dataset, model, par_buffer, device, max_sub):
    outputlogs_dir = BASE_DIR_GS + f'logs/{dataset}/'
    par_buffer_str = str(par_buffer).replace("'", '"')
    cmd_python = "from gridsearch import unsupervised\n" + f'unsupervised("{dataset}", "{model}", {par_buffer_str})'
    cmd_sbatch = "conda activate molspn\n" + f"python -c '{cmd_python}'"

    while True:
        run_squeue = subprocess.run(['squeue', f'--user={os.environ["USER"]}', '-h', '-r'], stdout=subprocess.PIPE)
        run_wcount = subprocess.run(['wc', '-l'], input=run_squeue.stdout, capture_output=True)
        num_queued = int(run_wcount.stdout)

        if len(par_buffer) <= max_sub - num_queued:
            if device == 'cuda':
                subprocess.run(['sbatch',
                                f'--job-name={model.replace("_","")}',
                                f'--output={outputlogs_dir}/{model}/%A_%a.out',
                                '--partition=amdgpudeadline',
                                '--ntasks=1',
                                '--mem-per-cpu=64000',
                                f'--gres=gpu:1',
                                f'--array=0-{len(par_buffer)-1}',
                                f'--wrap={cmd_sbatch}'])
            elif device == 'cpu':
                subprocess.run(['sbatch',
                                f'--job-name={model.replace("_","")}',
                                f'--output={outputlogs_dir}/{model}/%A_%a.out',
                                '--partition=amddeadline',
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
        'marg_sort',
        # 'zero_sort',
    ]
    gpu_models = MODELS.keys()

    # for dataset in MOLECULAR_DATASETS.keys():
    for dataset in ['qm9', 'zinc250k']:
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

            for hyperpars in gridsearch_hyperpars.GRIDS[model](dataset, model):
                hyperpars['model_hpars']['device'] = device
                backend_hpars_prefix(hyperpars)

                path = BASE_DIR_GS + f'eval/metrics/{dataset}/{model}/' + dict2str(flatten_dict(backend_hpars_prefix(hyperpars))) + '.csv'
                if not os.path.isfile(path):
                    par_buffer.append(hyperpars)

                if len(par_buffer) == max_jobs_to_submit:
                    submit_job(dataset, model, par_buffer, device, max_sub)
                    par_buffer = []

            if len(par_buffer) > 1:
                submit_job(dataset, model, par_buffer, device, max_sub)
                par_buffer = []

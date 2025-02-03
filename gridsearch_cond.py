import os
import re
import numpy as np
import time
import torch
import subprocess
import pandas as pd

from utils.datasets import MOLECULAR_DATASETS, BASE_DIR
from utils.conditional import evaluate_conditional

from rdkit import Chem, rdBase, RDLogger
rdBase.DisableLog("rdApp.error")

PATT_CONFIG = {
    'qm9': ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'C1=CC=CC=C1', 'C1CN1C', 'N1C=CC=C1', 'COC'],
    'zinc250k': ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'C1=CC=CC=C1', 'C1CN1C', 'N1C=CC=C1', 'COC']
}

BACKEND_NAMES = {
    'btree': 'BT',
    'vtree': 'LT',
    'rtree': 'RT',
    'ptree': 'RT-S',
    'ctree': 'HCLT'
}

IGNORE = [
    'device',
    'seed',
    'sam_valid',
    'sam_unique',
    'sam_unique_abs',
    'sam_novel',
    'sam_novel_abs',
    'sam_score',
    'sam_m_stab',
    'sam_a_stab',
    'sam_fcd_trn',
    'sam_kldiv_trn',
    'sam_nspdk_trn',
    'sam_fcd_val',
    'sam_kldiv_val',
    'sam_nspdk_val',
    'sam_fcd_tst',
    'sam_kldiv_tst',
    'sam_nspdk_tst',
    'res_valid',
    'res_unique',
    'res_unique_abs',
    'res_novel',
    'res_novel_abs',
    'res_score',
    'res_m_stab',
    'res_a_stab',
    'res_fcd_trn',
    'res_kldiv_trn',
    'res_nspdk_trn',
    'res_fcd_val',
    'res_kldiv_val',
    'res_nspdk_val',
    'res_fcd_tst',
    'res_kldiv_tst',
    'res_nspdk_tst',
    'cor_valid',
    'cor_unique',
    'cor_unique_abs',
    'cor_novel',
    'cor_novel_abs',
    'cor_score',
    'cor_m_stab',
    'cor_a_stab',
    'cor_fcd_trn',
    'cor_kldiv_trn',
    'cor_nspdk_trn',
    'cor_fcd_val',
    'cor_kldiv_val',
    'cor_nspdk_val',
    'cor_fcd_tst',
    'cor_kldiv_tst',
    'cor_nspdk_tst',
    'nll_trn_approx',
    'nll_val_approx',
    'nll_tst_approx',
    'time_sam',
    'time_res',
    'time_cor',
    'atom_list',
    'num_params',
    'model_path'
    ]


from models import molspn_zero
from models import molspn_marg

MODELS = {
    **molspn_zero.MODELS,
    **molspn_marg.MODELS
    }

BASE_DIR_COND = f'{BASE_DIR}cond/'


def find_best(evaluation_dir, dataset, model, backends):
    path_dict = {}
    path = evaluation_dir + f'metrics/{dataset}/{model}/'

    for i, backend in enumerate(backends.keys()):
        b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path) if backend in f])

        # b_frame = b_frame.loc[b_frame['nc'] == 256]
        g_frame = b_frame.groupby(list(filter(lambda x: x not in IGNORE, b_frame.columns)))
        a_frame = g_frame.agg({'sam_valid': 'mean'})
        f_frame = g_frame.get_group(a_frame['sam_valid'].idxmax())

        path_dict[backend] = list(f_frame['model_path'])

    return path_dict

def get_str_hpar(path, hpar_name):
    match = re.search(rf'{hpar_name}=([^\W_]+)', path)
    return str(match.group(1))

def get_num_hpar(path, hpar_name):
    match = re.search(rf'{hpar_name}=(\d+)', path)
    return int(match.group(1))

def cond_eval(dataset, name, path_buffer):
    torch.set_float32_matmul_precision('medium')
    RDLogger.DisableLog('rdApp.*')

    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']
    max_types = MOLECULAR_DATASETS[dataset]['max_types']

    path_model = path_buffer[int(os.environ["SLURM_ARRAY_TASK_ID"])]

    order = get_str_hpar(path_model, 'order')
    backend = get_str_hpar(path_model, 'backend')
    seed = get_num_hpar(path_model, 'seed')
    batch_size = get_num_hpar(path_model, 'batch_size')

    model = torch.load(path_model, weights_only=False)

    num_samples = 10000

    exp_info = {'dataset': dataset, 'model': name, 'backend': backend, 'order': order, 'seed': seed, 'num_samples': num_samples} 

    for patt in PATT_CONFIG[dataset]:
        metrics = evaluate_conditional(model, patt, dataset, max_atoms, atom_list, num_samples, batch_size=batch_size, seed=seed, order=order)
        metrics['pattern'] = patt
        metrics = metrics | exp_info
        df = pd.DataFrame.from_dict(metrics, 'index').transpose()
        print(df)

        dir = f'{BASE_DIR_COND}{dataset}/{name}/'
        os.makedirs(dir, exist_ok=True)
        path_cond = f'{dir}backend={backend}_seed={seed}_pattern={patt}'
        df.to_csv(path_cond + '.csv', index=False)

def submit_job(dataset, model, path_buffer, device, max_sub):
    outputlogs_dir = BASE_DIR_COND + f'logs/{dataset}/'
    path_buffer_str = str(path_buffer).replace("'", '"')
    cmd_python = "from gridsearch_cond import cond_eval\n" + f'cond_eval("{dataset}", "{model}", {path_buffer_str})'
    cmd_sbatch = "conda activate molspn\n" + f"python -c '{cmd_python}'"

    while True:
        run_squeue = subprocess.run(['squeue', f'--user={os.environ["USER"]}', '-h', '-r'], stdout=subprocess.PIPE)
        run_wcount = subprocess.run(['wc', '-l'], input=run_squeue.stdout, capture_output=True)
        num_queued = int(run_wcount.stdout)

        print(len(path_buffer), max_sub-num_queued)
        if len(path_buffer) <= max_sub - num_queued:
            if device == 'cuda':
                subprocess.run(['sbatch',
                                f'--job-name=cond{model.replace("_","")}',
                                f'--output={outputlogs_dir}/{model}/%A_%a.out',
                                '--partition=amdgpufast',
                                '--ntasks=1',
                                '--mem-per-cpu=64000',
                                f'--gres=gpu:1',
                                f'--array=0-{len(path_buffer)-1}',
                                f'--wrap={cmd_sbatch}'])
            elif device == 'cpu':
                subprocess.run(['sbatch',
                                f'--job-name=cond{model.replace("_","")}',
                                f'--output={outputlogs_dir}/{model}/%A_%a.out',
                                '--partition=amdfast',
                                '--ntasks=1',
                                '--ntasks-per-node=1',
                                '--cpus-per-task=1',
                                '--mem-per-cpu=64000',
                                f'--array=0-{len(path_buffer)-1}',
                                f'--wrap={cmd_sbatch}'])
            else:
                os.error('Unknown device.')

            break
        else:
            time.sleep(20)


if __name__ == "__main__":
    evaluation_dir = '/mnt/data/density_learning/molspn/gs0/eval/'


    path_buffer = []
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

            path_dict = find_best(evaluation_dir, dataset, model, BACKEND_NAMES)

            for backend in BACKEND_NAMES.keys():
                for path_model in path_dict[backend]:
                    path_buffer.append(path_model)
                
                if len(path_buffer) == max_jobs_to_submit:
                    submit_job(dataset, model, path_buffer, device, max_sub)
                    path_buffer = []
            
            if len(path_buffer) > 1:
                submit_job(dataset, model, path_buffer, device, max_sub)
                path_buffer = []

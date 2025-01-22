import os
import torch
import torch.optim as optim
import pandas as pd

from tqdm import tqdm
from timeit import default_timer
from rdkit.Chem.Draw import MolsToGridImage
from utils.graphs import unflatt_tril

from utils.evaluate import evaluate_molecules, resample_invalid_mols, count_parameters, print_metrics
from utils.molecular import correct_mols, mols2gs

IGNORED_HYPERPARS = [
    'atom_list',
    'optimizer'
]


def flatten_dict(d, input_key=''):
    if isinstance(d, dict):
        return {k if input_key else k: v for key, value in d.items() for k, v in flatten_dict(value, key).items()}
    else:
        return {input_key: d}

def dict2str(d):
    return '_'.join([f'{key}={value}' for key, value in d.items() if key not in IGNORED_HYPERPARS])

def backend_hpars_prefix(d):
    o = {}
    for key, value in d.items():
        match key:
            case 'bx_hpars':
                o[key] = {'x' + str(k): v for k, v in value.items()}
            case 'ba_hpars':
                o[key] = {'a' + str(k): v for k, v in value.items()}
            case _:
                if isinstance(value, dict):
                    o[key] = backend_hpars_prefix(value)
                else:
                    o[key] = value
    return o

def run_epoch(model, loader, optimizer=[], verbose=False):
    nll_sum = 0.
    for b in tqdm(loader, leave=False, disable=verbose):
        x = b['x'].to(model.device)
        a = b['a'].to(model.device)
        a = unflatt_tril(a, x.size(1))
        nll = -model.logpdf(x, a)
        nll_sum += nll
        if optimizer:
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

    return nll_sum.item() / len(loader)

METRIC_TYPES = ['valid', 'unique', 'novel', 'score']

def train(
        model,
        loaders,
        hyperpars,
        checkpoint_dir,
        num_nonimproving_epochs=2000,
        verbose=False,
        metric_type='score'
    ):
    # optimizer = optim.LBFGS(model.parameters(), **hyperpars['optimizer_hpars'], history_size=100, max_iter=5)
    optimizer = optim.Adam(model.parameters(), **hyperpars['optimizer_hpars'])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    lookahead_counter = num_nonimproving_epochs
    if metric_type in METRIC_TYPES:
        best_metric = 0e0
    else:
        best_metric = 1e6
    best_model_path = None
    save_model = False

    for epoch in range(hyperpars['num_epochs']):
        model.train()
        nll_trn = run_epoch(model, loaders['loader_trn'], verbose=verbose, optimizer=optimizer)
        # scheduler.step()
        model.eval()

        x_sam, a_sam = model.sample(1000)
        metrics = evaluate_molecules(x_sam, a_sam, loaders, hyperpars['atom_list'], metrics_only=True, canonical=(hyperpars['order']=='canonical'))
        metrics_str = print_metrics(metrics)

        if metric_type in METRIC_TYPES:
            metric = metrics[metric_type]
            print(f'epoch {epoch:3d}: ll_trn={-nll_trn:.4f}, ' + metrics_str)

            if metric >= best_metric:
                best_metric = metric
                lookahead_counter = num_nonimproving_epochs
                save_model = True
            else:
                lookahead_counter -= 1
        else:
            metric = run_epoch(model, loaders['loader_val'], verbose=verbose)
            print(f'epoch {epoch:3d}: ll_trn={-nll_trn:.4f}, ll_val={-metric:.4f}, ' + metrics_str)

            if metric < best_metric:
                best_metric = metric
                lookahead_counter = num_nonimproving_epochs
                save_model = True
            else:
                lookahead_counter -= 1

        if lookahead_counter == 0:
            break

        if save_model == True:
            dir = checkpoint_dir + f'{hyperpars["dataset"]}/{hyperpars["model"]}/'

            if os.path.isdir(dir) != True:
                os.makedirs(dir)
            if best_model_path != None:
                os.remove(best_model_path)
            path = dir + dict2str(flatten_dict(backend_hpars_prefix(hyperpars))) + '.pt'
            torch.save(model, path)
            best_model_path = path
            save_model == False

    return best_model_path

def evaluate(
        loaders,
        hyperpars,
        evaluation_dir,
        checkpoint_dir,
        num_samples=10000,
        compute_nll=True,
        verbose=False,
    ):
    dir = checkpoint_dir + f'{hyperpars["dataset"]}/{hyperpars["model"]}/'
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
        os.system('chmod -R 775 ' + dir)
    path_model = dir + dict2str(flatten_dict(backend_hpars_prefix(hyperpars))) + '.pt'
    model = torch.load(path_model, weights_only=False)
    model.eval()

    canonical = (hyperpars['order']=='canonical')
    atom_list = hyperpars['atom_list']

    start = default_timer()
    x_sam, a_sam = model.sample(num_samples)
    time_sam = default_timer() - start

    start = default_timer()
    x_res, a_res = resample_invalid_mols(model, num_samples, atom_list, x_sam.size(1), canonical=canonical)
    time_res = default_timer() - start

    start = default_timer()
    x_cor, a_cor = mols2gs(correct_mols(x_sam, a_sam, atom_list), x_sam.size(1), atom_list)
    time_cor = default_timer() - start

    with torch.no_grad():
        if compute_nll == True:
            nll_trn_approx = run_epoch(model, loaders['loader_trn'], verbose=verbose)
            nll_val_approx = run_epoch(model, loaders['loader_val'], verbose=verbose)
            nll_tst_approx = run_epoch(model, loaders['loader_tst'], verbose=verbose)
            metrics_neglogliks = {
                'nll_trn_approx': nll_trn_approx,
                'nll_val_approx': nll_val_approx,
                'nll_tst_approx': nll_tst_approx
            }
        else:
            metrics_neglogliks = {}

    mols_sam, _, metrics_sam = evaluate_molecules(x_sam, a_sam, loaders, atom_list, True, True, True, preffix='sam_', canonical=canonical)
    mols_res, _, metrics_res = evaluate_molecules(x_res, a_res, loaders, atom_list, True, True, True, preffix='res_', canonical=canonical)
    mols_cor, _, metrics_cor = evaluate_molecules(x_cor, a_cor, loaders, atom_list, True, True, True, preffix='cor_', canonical=canonical)

    metrics = {**metrics_sam,
               **metrics_res,
               **metrics_cor,
               **metrics_neglogliks,
               "time_sam": time_sam,
               "time_res": time_res,
               "time_cor": time_cor + time_sam,
               "num_params": count_parameters(model)}

    dir = evaluation_dir + f'metrics/{hyperpars["dataset"]}/{hyperpars["model"]}/'
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
        os.system('chmod -R 775 ' + dir)
    path_metrics = dir + dict2str(flatten_dict(backend_hpars_prefix(hyperpars)))
    df = pd.DataFrame.from_dict({**flatten_dict(backend_hpars_prefix(hyperpars)), **metrics}, 'index').transpose()
    df['model_path'] = path_model
    df.to_csv(path_metrics + '.csv', index=False)

    dir = evaluation_dir + f'images/{hyperpars["dataset"]}/{hyperpars["model"]}/'
    if os.path.isdir(dir) != True:
        os.makedirs(dir)
        os.system('chmod -R 775 ' + dir)
    path_images = dir + dict2str(flatten_dict(backend_hpars_prefix(hyperpars)))

    img_sam = MolsToGridImage(mols=mols_sam[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)
    img_res = MolsToGridImage(mols=mols_res[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)
    img_cor = MolsToGridImage(mols=mols_cor[0:64], molsPerRow=8, subImgSize=(200, 200), useSVG=False)

    img_sam.save(path_images + f'_san.png')
    img_res.save(path_images + f'_res.png')
    img_cor.save(path_images + f'_cor.png')

    return metrics

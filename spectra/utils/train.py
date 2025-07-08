import os
import torch
import torch.optim as optim

from tqdm import tqdm

from utils.evaluate import eval_metrics, eval_visual


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

def run_epoch(model, loader, optimizer=[], verbose=False):
    nll_sum = 0.
    n = 0 
    for b in tqdm(loader, leave=False, disable=verbose):
        x = b.float().to(model.device)
        nll = model(x).mean()
        nll_sum += nll * len(x)
        n += len(x)
        if optimizer:
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

    return nll_sum.item() / n


def train(
        model,
        loaders,
        hyperpars,
        base_dir,
        num_nonimproving_epochs=2000,
        verbose=False
    ):
    # optimizer = optim.LBFGS(model.parameters(), **hyperpars['optimizer_hpars'], history_size=100, max_iter=5)
    optimizer = optim.Adam(model.parameters(), **hyperpars['optimizer_hpars'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    lookahead_counter = num_nonimproving_epochs
    best_metric = float('inf')
    best_model_path = None
    save_model = False

    for epoch in range(hyperpars['num_epochs']):
        model.train()
        nll_trn = run_epoch(model, loaders['loader_trn'], verbose=verbose, optimizer=optimizer)
        scheduler.step()
        model.eval()

        metrics = eval_metrics(model, loaders)

        metric = run_epoch(model, loaders['loader_val'], verbose=verbose) # nll_val
        print(f'epoch {epoch:3d}: ll_trn={-nll_trn:.4f}, ll_val={-metric:.4f}, rse_trn={metrics["rse_trn"]:.3f}, rse_val={metrics["rse_val"]:.3f}')

        if metric < best_metric:
            best_metric = metric
            lookahead_counter = num_nonimproving_epochs
            save_model = True
        else:
            lookahead_counter -= 1

        if lookahead_counter == 0:
            break

        if save_model == True:
            dir = base_dir + f'ckpt/{hyperpars["dataset"]}/{hyperpars["model"]}/'
            os.makedirs(dir, exist_ok=True)
            if best_model_path != None:
                os.remove(best_model_path)
            path = dir + dict2str(flatten_dict(hyperpars)) + '.pt'
            torch.save(model, path)
            best_model_path = path
            save_model = False

    eval_visual(model, loaders['loader_tst'], loaders['transform'], hyperpars['model'])    

    return best_model_path

import os
import torch
import torch.optim as optim

from tqdm import tqdm

from utils.evaluate import rse


# from utils.evaluate import evaluate_molecules, resample_invalid_mols, count_parameters, print_metrics

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


def eval_metrics(model, loaders):
    t = loaders['transform']
    train_rse = []
    for b in loaders['loader_trn']:
        x = b.float().to(model.device)
        x_hat = model.reconstruct(x)
        train_rse.append(rse(t.inverse(x.to('cpu')), t.inverse(x_hat.to('cpu'))))
    train_rse = torch.cat(train_rse)
    train_rse = torch.mean(train_rse[train_rse<1])

    val_rse = []
    for b in loaders['loader_val']:
        x = b.float().to(model.device)
        x_hat = model.reconstruct(x)
        val_rse.append(rse(t.inverse(x.to('cpu')), t.inverse(x_hat.to('cpu'))))
    val_rse = torch.cat(val_rse)
    val_rse = torch.mean(val_rse[val_rse<1])

    return f"trn_rse: {train_rse:.3f}, val_rse: {val_rse:.3f}"

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
        print(f'epoch {epoch:3d}: ll_trn={-nll_trn:.4f}, ll_val={-metric:.4f}, ' +metrics)

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

    return best_model_path

import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import Optional

from utils.molecular import EMPTY_NODE_CAT, EMPTY_EDGE_CAT

def ffnn(ni: int,
        no: int,
        nl: int,
        batch_norm: bool,
        act = nn.ReLU(),
        final_act: Optional[any]=None
        ):
    nh = torch.arange(ni, no, (no - ni) / nl, dtype=torch.int)
    net = nn.Sequential()
    for i in range(len(nh) - 1):
        net.append(nn.Linear(nh[i], nh[i + 1]))
        net.append(act)
        if batch_norm:
            net.append(nn.BatchNorm1d(nh[i + 1]))
    net.append(nn.Linear(nh[-1], no))
    if final_act is not None:
        net.append(final_act)
    return net


def mask_g(x, a, mask):
    xm = x.clone()
    am = a.clone()

    xm[~mask] = EMPTY_NODE_CAT

    keep_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    am[~keep_mask] = EMPTY_EDGE_CAT
    return xm, am


class FFNNZeroSortMasked(nn.Module):
    def __init__(self, nd_n, nk_n, nk_e, nd_y, nl, device='cuda', min_sigma=1e-8):
        super(FFNNZeroSortMasked, self).__init__()
        self.nd_n = nd_n
        self.nd_l = int(nd_n * (nd_n-1) / 2)
        self.nk_n = nk_n
        self.nk_e = nk_e
        
        nd_g = nd_n * nk_n + self.nd_l * nk_e 
        self.net = ffnn(nd_g, nd_y, nl, batch_norm=True, act=nn.ReLU())
        self.net_mu = nn.Linear(nd_y, nd_y)
        self.sigma_params = nn.Parameter(torch.randn(1, nd_y))
        self.min_sigma = min_sigma
        self.m = torch.tril(torch.ones(nd_n, nd_n, dtype=torch.bool), diagonal=-1)

        self.to(device)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _flatten_graph(self, x: torch.Tensor, a: torch.Tensor):
        l = a[..., self.m].view(-1, self.nd_l)
        x_ohe = nn.functional.one_hot(x.long(), self.nk_n)
        l_ohe = nn.functional.one_hot(l.long(), self.nk_e)

        x_ohe_flat = x_ohe.view(len(x_ohe), -1)
        l_ohe_flat = l_ohe.view(len(l_ohe), -1)

        return torch.cat((x_ohe_flat, l_ohe_flat), dim=-1)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        g = self._flatten_graph(x, a).float()
        params = self.net(g)
        mu = self.net_mu(params)
        sigma = nn.functional.softplus(self.sigma_params) + self.min_sigma
        sigma = sigma.expand(len(mu), -1)
        return mu, sigma
    
    def create_mask(self, x, p=0.5):
        return torch.rand(x.shape) < p

    def objective(self, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor):
        m = self.create_mask(x)

        x, a = mask_g(x, a, m)

        mu, sigma = self(x, a)
        nll = -Normal(mu, sigma).log_prob(y).sum(-1)
        return nll
    
    @torch.no_grad
    def predict(self, x: torch.Tensor, a: torch.Tensor):
        return self(x, a)[0]

MODELS = {
    'ffnn_zero_sort_mask': FFNNZeroSortMasked
}


if __name__ == '__main__':
    import json

    with open(f'config/qm9/ffnn_zero_sort.json', 'r') as f:
        hyperpars = json.load(f)

    hps = hyperpars['model_hpars']
    model = FFNNZeroSortMasked(**hps)
    print(model)

    from utils.datasets import load_dataset

    loaders = load_dataset('qm9xas_canonical', 256, [0.8, 0.1, 0.1])

    batch = next(iter(loaders['loader_trn']))
    x, a, y = batch['x'].to(model.device), batch['a'].to(model.device), batch['spec'].to(model.device)

    model(x, a)

    model.objective(x, a, y)


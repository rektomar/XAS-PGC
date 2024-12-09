import os
import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from models.utils import ohe2cat, cat2ohe

from models.backend import backend_selector

from models.props import *


class MolSPNZeroSortProps(nn.Module):
    def __init__(self,
                 loader_trn,
                 hpars
                 ):
        super().__init__()
        x = torch.stack([b['x'] for b in loader_trn.dataset])
        a = torch.stack([b['a'] for b in loader_trn.dataset])

        network_x, nd_x, nk_x, network_a, nd_a, nk_a = backend_selector(x, a, hpars)

        self.nd_x = nd_x
        self.nd_a = nd_a
        self.nk_x = nk_x
        self.nk_a = nk_a

        self.network_x = network_x
        self.network_a = network_a
        self.network_y = PropNetwork(hpars['nc'])

        device = hpars['device']

        self.logits_n = nn.Parameter(torch.randn(          nd_x, device=device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.randn(1, hpars['nc'], device=device), requires_grad=True)
        self.m = torch.tril(torch.ones(self.nd_x, self.nd_x, dtype=torch.bool, device=device), diagonal=-1)

        self.to(device)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _forward(self, x, a, mx=None, ma=None):
        if ma is not None:
            ma = ma[:, self.m].view(-1, self.nd_a)
        self.network_x.set_marginalization_mask(mx)
        self.network_a.set_marginalization_mask(ma)

        n = (x > 0).sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)

        ll_card = dist_n.log_prob(n)
        ll_x = self.network_x(x)
        ll_a = self.network_a(a[:, self.m].view(-1, self.nd_a))

        return ll_card, ll_x, ll_a, torch.log_softmax(self.logits_w, dim=1)

    def forward(self, x, a, y):
        ll_card, ll_x, ll_a, ll_y, logits_w = self._forward(x, a)
        ll_y = self.network_y(y)
        return ll_card + torch.logsumexp(ll_x + ll_a + ll_y + logits_w, dim=1)

    def logpdf(self, x, a, y):
        return self(x, a, y).mean()

    def __sample(self, logits_w: torch.Tensor, logits_n: torch.Tensor, num_samples: int, xo=None, ao=None, mx=None, ma=None):
        if ma is not None:
            ma = ma[:, self.m].view(-1, self.nd_a)
        if ao is not None:
            ao = ao[:, self.m].view(-1, self.nd_a)
        self.network_x.set_marginalization_mask(mx)
        self.network_a.set_marginalization_mask(ma)

        dist_n = torch.distributions.Categorical(logits=logits_n)
        dist_w = torch.distributions.Categorical(logits=logits_w)
        samp_n = dist_n.sample((num_samples, ))
        samp_w = dist_w.sample((num_samples, )).squeeze(-1)

        mask_x = torch.arange(self.nd_x, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        mask_a = (mask_x.unsqueeze(2) * mask_x.unsqueeze(1))[:, self.m].view(-1, self.nd_a)

        x = self.network_x.sample(num_samples, class_idxs=samp_w, x=xo)
        l = self.network_a.sample(num_samples, class_idxs=samp_w, x=ao)
        x[~mask_x] = 0
        l[~mask_a] = 0

        a = torch.zeros((num_samples, self.nd_x, self.nd_x), device=self.device)
        a[:, self.m] = l
        a.transpose(1, 2)[:, self.m] = l

        return x.to(device='cpu', dtype=torch.int), a.to(device='cpu', dtype=torch.int)

    def _sample(self, num_samples):
        return self.__sample(self.logits_w, self.logits_n, num_samples)

    @torch.no_grad
    def sample(self, num_samples: int, chunk_size: int=2000):
        if num_samples > chunk_size:
            x_sam = []
            a_sam = []
            chunks = num_samples // chunk_size*[chunk_size] + ([num_samples % chunk_size] if num_samples % chunk_size > 0 else [])
            for n in chunks:
                x, a = self._sample(n)
                x_sam.append(x)
                a_sam.append(a)
            x_sam, a_sam = torch.cat(x_sam), torch.cat(a_sam)
        else:
            x_sam, a_sam = self._sample(num_samples)

        return x_sam, a_sam

    def sample_given_props(self, y: torch.Tensor, num_samples: int):
        # G ~ p(G | y), G = (X, A)
        # for y of shape [1, 1]   
        logits_w = self.network_y(y) + torch.log_softmax(self.logits_w, -1)
        x, a = self.__sample(logits_w, self.logits_n, num_samples)
        return x, a
        
    def logpdf_props(self, y: torch.Tensor):
        # log p(y)
        ll_y = self.network_y(y)
        return torch.logsumexp(ll_y + torch.log_softmax(self.logits_w, dim=1), dim=1)
    
    def predict_props(self, x: torch.Tensor, a: torch.Tensor, mx: torch.Tensor=None, ma: torch.Tensor=None):
        # returns E[y|G], where y are additional features/properties 
        # and G is graph induced by (x, a) and optional masks (mx, ma)

        _, ll_x, ll_a, logits_w = self._forward(x, a, mx=mx, ma=ma)

        logits_w = ll_x + ll_a + logits_w
        prior = Categorical(logits=logits_w)
        comps = self.network_y.distribution.distribution

        distr = MixtureSameFamily(prior, comps)
        return distr.mean


MODELS = {
    'zero_sort_props': MolSPNZeroSortProps,
}


if __name__ == '__main__':
    dataset = 'qm9'

    import json
    from utils.datasets import load_dataset

    with open(f'config/{dataset}/zero_sort_btree.json', 'r') as f:
        hyperpars = json.load(f)

    loaders = load_dataset(dataset, hyperpars['batch_size'], [0.8, 0.1, 0.1], order=hyperpars['order'])
    model = MolSPNZeroSortProps(loaders['loader_trn'], hyperpars['model_hpars'])

    model.sample(1)


    y = torch.rand(1, 1, device=model.device)
    model.logpdf_props(y)
    model.sample_given_props(y, 32)
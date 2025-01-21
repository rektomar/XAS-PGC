import torch
import torch.nn as nn

from models.backend import backend_selector
from typing import Optional


class MolSPNZeroSort(nn.Module):
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

        self.nc = hpars['nc']
        self.device = hpars['device']

        self.logits_n = nn.Parameter(torch.randn(nd_x,    device=self.device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.randn(self.nc, device=self.device), requires_grad=True)
        self.m = torch.tril(torch.ones(self.nd_x, self.nd_x, dtype=torch.bool, device=self.device), diagonal=-1)

        self.to(self.device)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        self.network_x.set_marginalization_mask(None)
        self.network_a.set_marginalization_mask(None)

        n = (x > 0).sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)

        ll_c = dist_n.log_prob(n)
        ll_x = self.network_x(x)
        ll_a = self.network_a(a[:, self.m].view(-1, self.nd_a))
        ll_w = torch.log_softmax(self.logits_w, dim=0).unsqueeze(0)

        return ll_c + torch.logsumexp(ll_x + ll_a + ll_w, dim=1)

    def logpdf(self, x, a):
        return self(x, a).mean()

    @torch.no_grad
    def _sample(self, num_samples: int=1, cond_x: Optional[torch.Tensor]=None, cond_a: Optional[torch.Tensor]=None):
        if cond_x is not None and cond_a is not None:
            if len(cond_x) == len(cond_a):
                num_samples = len(cond_x)
            else:
                raise 'len(cond_x) and len(cond_a) are not equal.'
        else:
            self.network_x.set_marginalization_mask(None)
            self.network_a.set_marginalization_mask(None)

        if cond_x is not None:
            mask_x = (cond_x > 0)
            logs_n = self.logits_n.unsqueeze(0).expand(num_samples, -1).masked_fill_(mask_x, -torch.inf)

            self.network_x.set_marginalization_mask(mask_x)
            logs_x = self.network_x(cond_x)
        else:
            mask_x = torch.zeros(num_samples, self.nd_x, device=self.device, dtype=torch.bool)
            logs_x = torch.zeros(num_samples, self.nc,   device=self.device)
            logs_n = self.logits_n.unsqueeze(0).expand(num_samples, -1)

        if cond_a is not None:
            mask_a = (cond_a > -1)
            mask_a = mask_a[:, self.m].view(-1, self.nd_a)
            cond_a = cond_a[:, self.m].view(-1, self.nd_a)

            self.network_a.set_marginalization_mask(mask_a)
            logs_a = self.network_a(cond_a)
        else:
            mask_a = torch.zeros(num_samples, self.nd_a, device=self.device, dtype=torch.bool)
            logs_a = torch.zeros(num_samples, self.nc,   device=self.device)

        logs_w = logs_x + logs_a + self.logits_w.unsqueeze(0)

        samp_n = torch.distributions.Categorical(logits=logs_n).sample()
        samp_w = torch.distributions.Categorical(logits=logs_w).sample()

        mask_xn = torch.arange(self.nd_x, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        mask_an = (mask_xn.unsqueeze(2) * mask_xn.unsqueeze(1))[:, self.m].view(-1, self.nd_a)

        x = self.network_x.sample(num_samples, class_idxs=samp_w, x=cond_x)
        l = self.network_a.sample(num_samples, class_idxs=samp_w, x=cond_a)

        x[~mask_xn] = 0
        l[~mask_an] = 0

        a = torch.zeros((num_samples, self.nd_x, self.nd_x), device=self.device)
        a[:, self.m] = l
        a.transpose(1, 2)[:, self.m] = l

        return x.to(device='cpu', dtype=torch.int), a.to(device='cpu', dtype=torch.int)

    @torch.no_grad
    def sample(self, num_samples: int=1, cond_x: Optional[torch.Tensor]=None, cond_a: Optional[torch.Tensor]=None, chunk_size: int=2000):
    
        if num_samples > chunk_size:
            x_sam = []
            a_sam = []
            chunks = num_samples // chunk_size*[chunk_size] + ([num_samples % chunk_size] if num_samples % chunk_size > 0 else [])
            for n in chunks:
                # TODO: fix chunking for conditional sampling
                x, a = self._sample(n, cond_x=cond_x, cond_a=cond_a)
                x_sam.append(x)
                a_sam.append(a)
            x_sam, a_sam = torch.cat(x_sam), torch.cat(a_sam)
        else:
            x_sam, a_sam = self._sample(num_samples, cond_x=cond_x, cond_a=cond_a)

        return x_sam, a_sam

MODELS = {
    'zero_sort': MolSPNZeroSort,
}


if __name__ == '__main__':
    dataset = 'zinc250k'

    import json
    from utils.datasets import load_dataset

    with open(f'config/{dataset}/zero_sort.json', 'r') as f:
        hyperpars = json.load(f)

    loader_trn, loader_val = load_dataset(dataset, 256, split=[0.8, 0.2], order='canonical')
    model = MolSPNZeroSort(loader_trn, hyperpars['model_hpars'])

    model.sample(1)

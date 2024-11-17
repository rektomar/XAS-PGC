import torch
import torch.nn as nn

from models.backend import backend_selector


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

        device = hpars['device']

        self.logits_n = nn.Parameter(torch.randn(          nd_x, device=device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.randn(1, hpars['nc'], device=device), requires_grad=True)
        self.m = torch.tril(torch.ones(self.nd_x, self.nd_x, dtype=torch.bool, device=device), diagonal=-1)

        self.device = device
        self.to(device)

    def forward(self, x, a):
        n = (x > 0).sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)

        ll_x = self.network_x(x)
        ll_a = self.network_a(a[:, self.m].view(-1, self.nd_a))

        return dist_n.log_prob(n) + torch.logsumexp(ll_x + ll_a + torch.log_softmax(self.logits_w, dim=1), dim=1)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        dist_n = torch.distributions.Categorical(logits=self.logits_n)
        dist_w = torch.distributions.Categorical(logits=self.logits_w)
        samp_n = dist_n.sample((num_samples, ))
        samp_w = dist_w.sample((num_samples, )).squeeze()

        mask_x = torch.arange(self.nd_x, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        mask_a = (mask_x.unsqueeze(2) * mask_x.unsqueeze(1))[:, self.m].view(-1, self.nd_a)

        x = self.network_x.sample(num_samples, class_idxs=samp_w)
        l = self.network_a.sample(num_samples, class_idxs=samp_w)
        x[~mask_x] = 0
        l[~mask_a] = 0

        a = torch.zeros((num_samples, self.nd_x, self.nd_x), device=self.device)
        a[:, self.m] = l
        a.transpose(1, 2)[:, self.m] = l

        return x.cpu(), a.cpu()

MODELS = {
    'zero_sort': MolSPNZeroSort,
}

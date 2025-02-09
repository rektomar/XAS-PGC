import torch
import torch.nn as nn

from torch.distributions import Normal, Categorical, MixtureSameFamily, Beta
from torch.nn.functional import softplus

# TODO: merge prop models using abstract class


class NormalProp(nn.Module):

    def __init__(self, nc: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.m = nn.Parameter(torch.randn(nc))
        self.s = nn.Parameter(torch.randn(nc))

    @property
    def distribution(self):
        return Normal(self.m, softplus(self.s)+self.eps)
    
    def logpdf(self, y):
        return self.distribution.log_prob(y)
    
    def forward(self, y):
        return self.logpdf(y)
    
    def sample(self, n_samples: int):
        return self.distribution.sample((n_samples,))
    

class GMMProp(nn.Module):

    def __init__(self, nc: int, num_comps: int=2, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.m = nn.Parameter(torch.randn(nc, num_comps))
        self.s = nn.Parameter(torch.randn(nc, num_comps))
        self.w = nn.Parameter(torch.randn(nc, num_comps))

    @property
    def distribution(self):
        mix = Categorical(logits=self.w)
        comp = Normal(self.m, softplus(self.s)+self.eps)
        return MixtureSameFamily(mix, comp)
    
    def logpdf(self, y):
        return self.distribution.log_prob(y)
    
    def forward(self, y):
        return self.logpdf(y)
    
    def sample(self, n_samples: int):
        return self.distribution.sample((n_samples,))


class BetaProp(nn.Module):

    def __init__(self, nc: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.randn(nc))
        self.beta = nn.Parameter(torch.randn(nc))

    @property
    def distribution(self):
        return Beta(softplus(self.alpha)+self.eps, softplus(self.beta)+self.eps)
    
    def logpdf(self, y):
        return self.distribution.log_prob(y)
    
    def forward(self, y):
        return self.logpdf(y)
    
    def sample(self, n_samples: int):
        return self.distribution.sample((n_samples,))


class PropNetwork(nn.Module):

    def __init__(self, nc: int, prop_type: str):
        super().__init__()
        if prop_type in PROP_DISTR_CONFIG.keys():
            self.prop_dist = PROP_DISTR_CONFIG[prop_type](nc)
        else:
            raise f'Unknown selected prop {prop_type},'

    def forward(self, y):
        return self.prop_dist.logpdf(y)
        
    def sample(self, n_samples):
        print(self.prop_dist.distribution)
        return self.prop_dist.sample(n_samples)

PROP_DISTR_CONFIG = {
    'logP': NormalProp,
    'MW'  : NormalProp,
    'QED' : BetaProp,
}


if __name__ == '__main__':
    model = BetaProp(10)
    y_s = model.sample(100)
    y = torch.rand(64, 1)
    model(y)

    model = BetaProp(10)
    y_s = model.sample(100)
    y = torch.rand(64, 1)
    model(y)

    model = GMMProp(10, 3)
    y_s = model.sample(100)
    y = torch.rand(64, 1)
    model(y)

    PropNetwork(10, 'logP')

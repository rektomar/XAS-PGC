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

    def __init__(self, nc: int, num_comps: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.m = nn.Parameter(torch.randn(nc, num_comps))
        self.s = nn.Parameter(torch.randn(nc, num_comps))
        self.w = nn.Parameter(torch.randn(num_comps))

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

    def __init__(self, nc: int):
        super().__init__()

        # nd = 1
        # num_comps = 4
        # self.distribution = GMMProp(nc, num_comps)

        self.distribution = NormalProp(nc)

    def forward(self, y):
        return self.distribution.logpdf(y)
        
    def sample(self, n_samples):
        return self.distribution.sample(n_samples)


if __name__ == '__main__':
    model = BetaProp(10)
    model.sample(100)
    y = torch.rand(64, 1)
    model(y)

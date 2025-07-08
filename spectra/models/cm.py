import math
import torch
import torch.nn as nn
# import qmcpy

from torch.distributions import Categorical, Normal, LogNormal
from typing import Callable, Optional, List
from torch import Tensor

# the continuous mixture model is based on the following implementation: https://github.com/AlCorreia/cm-tpm

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

def normal_parse_params(mu, sigma_params, min_sigma=1e-8):
    sigma = nn.functional.softplus(sigma_params) + min_sigma
    return Normal(mu, sigma)

def lognormal_parse_params(mu, sigma_params, min_sigma=1e-8):
    sigma = nn.functional.softplus(sigma_params) + min_sigma
    return LogNormal(mu, sigma)


class BackFFNN(nn.Module):
    def __init__(self, nd_x, nd_z, nl, min_sigma=1e-8, device: Optional[str]='cuda'):
        super(BackFFNN, self).__init__()
        self.net = ffnn(nd_z, nd_x, nl, batch_norm=True, act=nn.ReLU())
        self.net_mu = nn.Linear(nd_x, nd_x)
        self.net_sigma = nn.Linear(nd_x, nd_x)
        self.min_sigma = min_sigma

        self.to(device)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self, z):                                    
        params = self.net(z)
        m, s = self.net_mu(params), self.net_sigma(params)
        return m, torch.nn.functional.softplus(s)+self.min_sigma

class GaussianSampler:
    def __init__(self,
                 nd: int,
                 num_samples: int,
                 mean: Optional[float]=0.0,
                 sdev: Optional[float]=10.0,
                 device: Optional[str]='cuda'
                 ):
        self.nd = nd
        self.num_samples = num_samples
        self.mean = mean
        self.sdev = sdev
        self.device = device

    def __call__(self, seed: Optional[int]=None, dtype=torch.float32):
        z = self.sdev*torch.randn(self.num_samples, self.nd, dtype=dtype, device=self.device)
        w = torch.full((self.num_samples,), math.log(1 / self.num_samples), dtype=dtype, device=self.device)
        return z, w

# class GaussianQMCSampler:
#     def __init__(self,
#                  nd: int,
#                  num_samples: int,
#                  mean: Optional[float]=0.0,
#                  covariance: Optional[float]=1.0
#                  ):
#         self.nd = nd
#         self.num_samples = num_samples
#         self.mean = mean
#         self.covariance = covariance

#     def __call__(self, seed: Optional[int]=None, dtype=torch.float32):
#         if seed is None:
#             seed = torch.randint(10000, (1,)).item()
#         latt = qmcpy.Lattice(dimension=self.nd, randomize=True, seed=seed)
#         dist = qmcpy.Gaussian(sampler=latt, mean=self.mean, covariance=self.covariance)
#         z = torch.from_numpy(dist.gen_samples(self.num_samples))
#         log_w = torch.full(size=(self.num_samples,), fill_value=math.log(1 / self.num_samples))
#         return z.type(dtype), log_w.type(dtype)


class Decoder(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 eps: float = 1e-8,
                 device: Optional[str]='cuda'
                 ):
        super(Decoder, self).__init__()
        self.network = network
        self.eps = eps
        self.to(device)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self, x: torch.Tensor, z: torch.Tensor, num_chunks: Optional[int]=None):
        x = x.unsqueeze(1).float()                           
                    
        log_prob = torch.zeros(len(x), len(z), device=self.device)     # (batch_size, num_chunks*chunk_size)
        for c in torch.arange(len(z)).chunk(num_chunks):
            m, s = self.network(z[c, :].to(self.device))
            log_prob_x = Normal(m, s).log_prob(x) # (batch_size, chunk_size, nd_node)
            log_prob[:, c] = log_prob_x.sum(dim=2)

        return log_prob
    
    def mean(self, z: torch.Tensor, num_chunks: Optional[int]=None):

        mean = torch.zeros(100, len(z), device=self.device)     # (batch_size, num_chunks*chunk_size)
        for c in torch.arange(len(z)).chunk(num_chunks):
            m, _ = self.network(z[c, :].to(self.device))
            mean[:, c] = torch.transpose(m, 0, 1)

        return mean

    @torch.no_grad
    def sample(self, z: torch.Tensor, k: torch.Tensor):
        m, s = self.network(z[k].to(self.device))
        return Normal(m, s).sample()


class ContinuousMixture(nn.Module):
    def __init__(self,
                 decoder: nn.Module,
                 sampler: Callable,
                 num_chunks: Optional[int]=2,
                 device: Optional[str]='cuda'
                 ):
        super(ContinuousMixture, self).__init__()
        self.decoder = decoder
        self.sampler = sampler
        self.num_chunks = num_chunks
        
        self.to(device)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self,
                x: torch.Tensor,
                z: Optional[torch.Tensor]=None,
                log_w: Optional[torch.Tensor]=None,
                seed: Optional[int]=None
                ):
        if z is None:
            z, log_w = self.sampler(seed=seed)
        log_prob = self.decoder(x, z, self.num_chunks)
        log_prob = torch.logsumexp(log_prob + log_w.to(self.device).unsqueeze(0), dim=1)
        return log_prob

    @torch.no_grad
    def sample(self, num_samples: int):
        z, log_w = self.sampler()
        k = Categorical(logits=log_w).sample([num_samples])
        return self.decoder.sample(z, k)
    
    @torch.no_grad
    def mean(self, z: Optional[torch.Tensor]=None, log_w: Optional[torch.Tensor]=None, seed: Optional[int]=None):
        if z is None:
            z, log_w = self.sampler(seed=seed)
        m = self.decoder.mean(z, self.num_chunks) 

        return torch.sum(m * torch.softmax(log_w, 0), 1)


class CM(nn.Module):
    def __init__(self, nd_x, nd_z, nl, nb, nc, distr='normal', device='cpu'):
        super(CM, self).__init__()

        backbone = BackFFNN(nd_x, nd_z, nl)
        self.network = ContinuousMixture(
            decoder=Decoder(backbone, device=device),
            sampler=GaussianSampler(nd_z, nb, device=device),
            num_chunks=nc,
            device=device
        )

        self.to(device)

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    def objective(self, x: Tensor) -> Tensor:
        return -self.network(x)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.objective(x)
    
    @torch.no_grad
    def sample(self, num_samples: int) -> Tensor:
        return self.network.sample(num_samples)
    
    @torch.no_grad()
    def reconstruct(self, x: Tensor):
        return self.network.mean().unsqueeze(0).expand(len(x), -1)

MODELS = {
    'cm': CM,
}


if __name__ == '__main__':
    
    model = CM(100, 64, 2, 2**13, 2)
    x = model.sample(32)
    print(x.shape)

    ll = model(x)
    print(ll, ll.shape)

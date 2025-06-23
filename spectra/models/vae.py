import torch
from torch import nn, Tensor
from torch.distributions import Normal, LogNormal, kl_divergence

from math import log

from typing import Optional


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


class Encoder(nn.Module):
    def __init__(self, nd_x, nd_z, nl, min_sigma=1e-8):
        super(Encoder, self).__init__()
        self.net = ffnn(nd_x, nd_z, nl, batch_norm=True, act=nn.ReLU())
        self.net_mu = nn.Linear(nd_z, nd_z)
        self.net_sigma = nn.Linear(nd_z, nd_z)
        self.min_sigma = min_sigma
 
    def forward(self, x: Tensor):
        params = self.net(x)
        mu, sigma_params = self.net_mu(params), self.net_sigma(params)
        return normal_parse_params(mu, sigma_params, self.min_sigma)
    

class Decoder(nn.Module):
    def __init__(self, nd_x, nd_z, nl, distr='normal', min_sigma=1e-8):
        super(Decoder, self).__init__()
        self.net = ffnn(nd_z, nd_x, nl, batch_norm=True, act=nn.ReLU())
        self.net_mu = nn.Linear(nd_x, nd_x)
        self.net_sigma = nn.Linear(nd_x, nd_x)
        self.min_sigma = min_sigma

        if distr == 'normal':
            self.distr_parse = normal_parse_params
        elif distr == 'lognormal':
            self.distr_parse = normal_parse_params
        else:
            raise f'{distr} not supported'
 
    def forward(self, x: Tensor):
        params = self.net(x)
        mu, sigma_params = self.net_mu(params), self.net_sigma(params)
        return self.distr_parse(mu, sigma_params, self.min_sigma)


class VAE(nn.Module):
    def __init__(self, nd_x, nd_z, nl, distr='normal', num_is=1000, beta=1, device='cpu'):
        super(VAE, self).__init__()

        self.encoder = Encoder(nd_x, nd_z, nl)
        self.decoder = Decoder(nd_x, nd_z, nl, distr=distr)

        self.prior = Normal(torch.zeros(nd_z, device=device), torch.ones(nd_z, device=device))
        self.beta = beta  # KL div weight in ELBO
        self.num_importance_samples = num_is

        self.to(device)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _objective(self, x: Tensor):
        posterior = self.encoder(x) # q(z|x)
        kl_div = kl_divergence(posterior, self.prior).sum(-1)

        z = posterior.rsample()
        generative = self.decoder(z)  # p(x|z)
        rec_ll = generative.log_prob(x).sum(-1)

        return rec_ll, kl_div
    
    def objective(self, x: Tensor) -> Tensor:
        rec_ll, kl_div = self._objective(x)

        elbo = rec_ll - self.beta * kl_div
        return -elbo
    
    def forward(self, x: Tensor):
        return self.objective(x)
    
    @torch.no_grad
    def logpdf(self, x: Tensor) -> Tensor:
        posterior = self.encoder(x)
        z = posterior.sample((self.num_importance_samples,))

        prior_ll = self.prior.log_prob(z).sum(-1)
        posterior_ll = posterior.log_prob(z).sum(-1)
        rec_ll = self.decoder(z).log_prob(x).sum(-1)
        
        ll = torch.logsumexp(rec_ll + prior_ll - posterior_ll, 0) - log(self.num_importance_samples)
        return ll
    
    @torch.no_grad
    def sample(self, num_samples: int) -> Tensor:
        z = self.prior.sample((num_samples, ))
        samples = self.decoder(z).sample()
        return samples
    
    @torch.no_grad()
    def reconstruct(self, x: Tensor):
        posterior = self.encoder(x) # q(z|x)
        z = posterior.sample()
        generative = self.decoder(z)  # p(x|z)
        return generative.sample()
        
MODELS = {
    'vae': VAE
}

if __name__ == '__main__':
    model = VAE(100, 32, 4)
    x = model.sample(64)
    print(x.shape)

    ll = model(x)
    print(ll.shape)

    rec = model.reconstruct(x)
    print(rec.shape)



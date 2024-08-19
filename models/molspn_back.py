import os
import math
import torch
import torch.nn as nn
import qmcpy

from torch.distributions import Categorical
from typing import Callable, Optional
from abc import abstractmethod
from models.spn_utils import ohe2cat, cat2ohe

# the continuous mixture model is based on the following implementation: https://github.com/AlCorreia/cm-tpm

class MLPSPlit(nn.Module):
    def __init__(self,
                 nd_node: int,
                 nk_node: int,
                 nk_edge: int,
                 nz_back: int,
                 nh_back: int,
                 nl_back: int,
                 device: Optional[str]='cuda'
                 ):
        super(MLPSPlit, self).__init__()
        nd_edge = nd_node * (nd_node - 1) // 2

        self.nd_node = nd_node
        self.nd_edge = nd_edge
        self.nk_node = nk_node
        self.nk_edge = nk_edge
        self.net_back = self._mlp_decoder(nz_back,            nh_back,  nl_back, True)
        self.net_node = self._mlp_decoder(nh_back//2, nd_node*nk_node,        2, False)
        self.net_edge = self._mlp_decoder(nh_back//2, nd_edge*nk_edge,        2, False)
        self.device = device

    def forward(self, z):
        h_back = self.net_back(z)
        h_node, h_edge = torch.chunk(h_back, 2, 1)
        h_node = self.net_node(h_node)
        h_edge = self.net_edge(h_edge)
        return h_node, h_edge

    @staticmethod
    def _mlp_decoder(ni: int,
                     no: int,
                     nl: int,
                     batch_norm: bool,
                     final_act: Optional[any]=None
                     ):
        nh = torch.arange(ni, no, (no - ni) / nl, dtype=torch.int)
        decoder = nn.Sequential()
        for i in range(len(nh) - 1):
            decoder.append(nn.Linear(nh[i], nh[i + 1]))
            decoder.append(nn.ReLU())
            if batch_norm:
                decoder.append(nn.BatchNorm1d(nh[i + 1]))
        decoder.append(nn.Linear(nh[-1], no))
        if final_act is not None:
            decoder.append(final_act)
        return decoder


class GaussianQMCSampler:
    def __init__(self,
                 nd: int,
                 num_samples: int,
                 mean: Optional[float]=0.0,
                 covariance: Optional[float]=1.0
                 ):
        self.nd = nd
        self.num_samples = num_samples
        self.mean = mean
        self.covariance = covariance

    def __call__(self, seed: Optional[int]=None, dtype=torch.float32):
        if seed is None:
            seed = torch.randint(10000, (1,)).item()
        latt = qmcpy.Lattice(dimension=self.nd, randomize=True, seed=seed)
        dist = qmcpy.Gaussian(sampler=latt, mean=self.mean, covariance=self.covariance)
        z = torch.from_numpy(dist.gen_samples(self.num_samples))
        log_w = torch.full(size=(self.num_samples,), fill_value=math.log(1 / self.num_samples))
        return z.type(dtype), log_w.type(dtype)


class CategoricalDecoder(nn.Module):
    def __init__(self,
                 nd_node: int,
                 nk_node: int,
                 nk_edge: int,
                 network: nn.Module,
                 device: Optional[str]='cuda'
                 ):
        super(CategoricalDecoder, self).__init__()
        self.nd_node = nd_node
        self.nd_edge = nd_node * (nd_node - 1) // 2
        self.nk_node = nk_node
        self.nk_edge = nk_edge
        self.network = network
        self.device = device

    def forward(self,
                x_node: torch.Tensor,
                x_edge: torch.Tensor,
                z: torch.Tensor,
                num_chunks: Optional[int]=None
                ):
        i_chunks = torch.arange(len(z)).chunk(num_chunks)
        log_prob = torch.zeros(len(x_node), len(z), device=self.device)
        x_edge = x_edge.view(-1, self.nd_edge)
        x_node = x_node.unsqueeze(1).float()
        x_edge = x_edge.unsqueeze(1).float()
        for i_chunk in i_chunks:
            logit_node, logit_edge = self.network(z[i_chunk, :].to(self.device)) # (chunk_size, nd_nodes * nk_nodes), (chunk_size, nd_nodes^2 * nk_edges)
            logit_node = logit_node.view(-1, self.nd_node, self.nk_node) # (chunk_size, nd_nodes,   nk_nodes)
            logit_edge = logit_edge.view(-1, self.nd_edge, self.nk_edge) # (chunk_size, nd_nodes^2, nk_edges)
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node).sum(dim=2) # (batch_size, chunk_size, nd_nodes)
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge).sum(dim=2) # (batch_size, chunk_size, nd_nodes^2)
            log_prob[:, i_chunk] = log_prob_node + log_prob_edge

        return log_prob

    def sample(self, z: torch.Tensor, k: torch.Tensor):
        logit_node, logit_edge = self.network(z[k].to(self.device))
        logit_node = logit_node.view(-1, self.nd_node, self.nk_node)
        logit_edge = logit_edge.view(-1, self.nd_edge, self.nk_edge)
        x_node = Categorical(logits=logit_node).sample()
        x_edge = Categorical(logits=logit_edge).sample()
        return x_node, x_edge


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
        self.device = device

    def forward(self,
                x_node: torch.Tensor,
                x_edge: torch.Tensor,
                z: Optional[torch.Tensor]=None,
                log_w: Optional[torch.Tensor]=None,
                seed: Optional[int]=None
                ):
        if z is None:
            z, log_w = self.sampler(seed=seed)
        log_prob = self.decoder(x_node, x_edge, z, self.num_chunks)
        log_prob = torch.logsumexp(log_prob + log_w.to(self.device).unsqueeze(0), dim=1)
        return log_prob

    def sample(self, num_samples: int):
        z, log_w = self.sampler()
        k = Categorical(logits=log_w).sample([num_samples])
        x_node, x_edge = self.decoder.sample(z, k)
        return x_node, x_edge


class MolSPNBackCore(nn.Module):
    def __init__(self,
                 nd_n: int,
                 nk_n: int,
                 nk_e: int,
                 nz: int,
                 nh: int,
                 nl: int,
                 nb: int,
                 nc: int,
                 device: str,
                 regime: str
                 ):
        super().__init__()
        nd_e = nd_n * (nd_n - 1) // 2

        self.nd_node = nd_n
        self.nd_edge = nd_e
        self.nk_node = nk_n
        self.nk_edge = nk_e
        self.regime = regime

        backbone = MLPSPlit(nd_n, nk_n, nk_e, nz, nh, nl)
        self.network = ContinuousMixture(
            decoder=CategoricalDecoder(nd_n, nk_n, nk_e, backbone),
            sampler=GaussianQMCSampler(nz, nb),
            num_chunks=nc,
            device=device
        )

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, x, a):
        pass

    def forward(self, x, a):
        x = x.to(self.device)
        a = a.to(self.device)
        if   self.regime == 'cat':
            _x, _a = ohe2cat(x, a)
        elif self.regime == 'deq':
            _x = x + self.dc_n*torch.rand(x.size(), device=self.device)
            _a = a + self.dc_e*torch.rand(a.size(), device=self.device)
        else:
            os.error('Unknown regime')
        return self._forward(_x, _a)

    def logpdf(self, x, a):
        return self(x, a).mean()

    @abstractmethod
    def _sample(self, num_samples):
        pass

    def sample(self, num_samples):
        x, a = self._sample(num_samples)
        if self.regime == 'cat':
            x, a = cat2ohe(x, a, self.nk_node, self.nk_edge)
        return x, a


class MolSPNBackSort(MolSPNBackCore):
    def __init__(self,
                 nd_n: int,
                 nk_n: int,
                 nk_e: int,
                 nz: int,
                 nh: int,
                 nl: int,
                 nb: int,
                 nc: int,
                 device: Optional[str]='cuda',
                 regime: Optional[str]='cat',
                 ):
        super().__init__(nd_n, nk_n, nk_e, nz, nh, nl, nb, nc, device, regime)

    def _forward(self, x, a):
        if   self.regime == 'cat':
            m = torch.tril(torch.ones(len(x), self.nd_node, self.nd_node, dtype=torch.bool), diagonal=-1)
            l = a[m].view(-1, self.nd_edge)
        elif self.regime == 'deq':
            m = torch.tril(torch.ones(len(x), self.nk_edge, self.nd_node, self.nd_node, dtype=torch.bool), diagonal=-1)
            l = a[m].view(-1, self.nk_edge, self.nd_edge)
        else:
            os.error('Unknown regime')
        return self.network(x, l)

    def _sample(self, num_samples):
        x, l = self.network.sample(num_samples)
        x = x.cpu()
        l = l.cpu()

        if   self.regime == 'cat':
            m = torch.tril(torch.ones(num_samples, self.nd_node, self.nd_node, dtype=torch.bool), diagonal=-1)
            a = torch.zeros(num_samples, self.nd_node, self.nd_node, dtype=torch.long)
            a[m] = l.view(num_samples*self.nd_edge)
        elif self.regime == 'deq':
            m = torch.tril(torch.ones(num_samples, self.nk_edge, self.nd_node, self.nd_node, dtype=torch.bool), diagonal=-1)
            a = torch.zeros(num_samples, self.nk_edge, self.nd_node, self.nd_node)
            l = torch.movedim(l, -1, 1)
            a[m] = l.reshape(num_samples*self.nd_edge*self.nk_edge)
        else:
            os.error('Unknown regime')

        return x, a


MODELS = {
    'molspn_back_sort': MolSPNBackSort,
}

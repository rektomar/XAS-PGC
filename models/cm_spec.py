import math
import torch
import torch.nn as nn
# import qmcpy

from torch.distributions import Categorical, Normal
from typing import Callable, Optional, List

# the continuous mixture model is based on the following implementation: https://github.com/AlCorreia/cm-tpm

def ffnn_decoder(ni: int,
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


class BackFFNN(nn.Module):
    def __init__(self,
                 nd_node: int,
                 nd_edge: int,
                 nd_spec: int,
                 nk_node: int,
                 nk_edge: int,
                 nz_back: int,
                 nh_back: int,
                 nl_back: int,
                 nl_node: int,
                 nl_edge: int,
                 nl_spec: int,
                 device: Optional[str]='cuda'
                 ):
        super(BackFFNN, self).__init__()

        self.nd_node = nd_node
        self.nd_edge = nd_edge
        self.nd_spec = nd_spec
        self.nk_node = nk_node
        self.nk_edge = nk_edge
        self.net_back = ffnn_decoder(nz_back, nh_back,  nl_back, True )
        self.net_node = ffnn_decoder(nh_back, nd_node*nk_node,  nl_node, False)
        self.net_edge = ffnn_decoder(nh_back, nd_edge*nk_edge,  nl_edge, False)
        self.net_spec = ffnn_decoder(nh_back, 2*nd_spec, nl_spec, False) 
        self.device = device

    def forward(self, z):                                    
        h_back = self.net_back(z)                           
        h_node = self.net_node(h_back)                       
        h_edge = self.net_edge(h_back)                       
        h_spec = self.net_spec(h_back)

        h_node = h_node.view(-1, self.nd_node, self.nk_node) 
        h_edge = h_edge.view(-1, self.nd_edge, self.nk_edge) 
        h_spec_m, h_spec_logs = torch.chunk(h_spec, 2, 1)
        return h_node, h_edge, h_spec_m, h_spec_logs


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


class HybridDecoder(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 eps: float = 1e-8,
                 device: Optional[str]='cuda'
                 ):
        super(HybridDecoder, self).__init__()
        self.network = network
        self.device = device
        self.eps = eps

    @torch.no_grad
    def forward_spec(self, x_spec: torch.Tensor, z: torch.Tensor, num_chunks: Optional[int]=None):
        # helper method for infenrence tasks
        x_spec = x_spec.unsqueeze(1).float()                                      

        log_prob = torch.zeros(len(x_spec), len(z), device=self.device)     # (batch_size, num_chunks*chunk_size)
        for c in torch.arange(len(z)).chunk(num_chunks):
            _, _, m_spec, logs_spec = self.network(z[c, :].to(self.device))  # (chunk_size, nd_node, nk_node), (chunk_size, nd_edge, nk_edge)
            log_prob[:, c] = Normal(m_spec, torch.nn.functional.softplus(logs_spec)+1e-8).log_prob(x_spec).sum(dim=2)

        return log_prob
    
    @torch.no_grad
    def distr_spec(self, z: torch.Tensor, num_chunks: Optional[int]=None):
        # helper method for infenrence tasks
        m_specs = torch.zeros(len(z), self.network.nd_spec, device=self.device)  
        s_specs = torch.zeros(len(z), self.network.nd_spec, device=self.device)
        for c in torch.arange(len(z)).chunk(num_chunks):
            _, _, m_spec, logs_spec = self.network(z[c, :].to(self.device))  # (chunk_size, nd_node, nk_node), (chunk_size, nd_edge, nk_edge)
            m_specs[c] = m_spec
            s_specs[c] = torch.nn.functional.softplus(logs_spec)+1e-8
        return m_specs, s_specs
    
    @torch.no_grad
    def forward_graph(self, x_node: torch.Tensor, x_edge: torch.Tensor, z: torch.Tensor, num_chunks: Optional[int]=None):
        # helper method for infenrence tasks
        x_node = x_node.unsqueeze(1).float()                           
        x_edge = x_edge.unsqueeze(1).float()                                 

        log_prob = torch.zeros(len(x_node), len(z), device=self.device)     # (batch_size, num_chunks*chunk_size)
        for c in torch.arange(len(z)).chunk(num_chunks):
            logit_node, logit_edge, _, _ = self.network(z[c, :].to(self.device))  # (chunk_size, nd_node, nk_node), (chunk_size, nd_edge, nk_edge)
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node) # (batch_size, chunk_size, nd_node)
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge) # (batch_size, chunk_size, nd_edge)
            log_prob[:, c] = log_prob_node.sum(dim=2) + log_prob_edge.sum(dim=2)

        return log_prob

    def forward(self, x_node: torch.Tensor, x_edge: torch.Tensor, x_spec: torch.Tensor, z: torch.Tensor, num_chunks: Optional[int]=None):
        x_node = x_node.unsqueeze(1).float()                           
        x_edge = x_edge.unsqueeze(1).float()
        x_spec = x_spec.unsqueeze(1).float()                                      

        log_prob = torch.zeros(len(x_node), len(z), device=self.device)     # (batch_size, num_chunks*chunk_size)
        for c in torch.arange(len(z)).chunk(num_chunks):
            logit_node, logit_edge, m_spec, logs_spec = self.network(z[c, :].to(self.device))
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node) # (batch_size, chunk_size, nd_node)
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge) # (batch_size, chunk_size, nd_edge)
            log_prob_spec = Normal(m_spec, torch.nn.functional.softplus(logs_spec)+1e-8).log_prob(x_spec)  # (batch_size, chunk_size, 100)
            log_prob[:, c] = log_prob_node.sum(dim=2) + log_prob_edge.sum(dim=2) + log_prob_spec.sum(dim=2)

        return log_prob

    @torch.no_grad
    def sample(self, z: torch.Tensor, k: torch.Tensor):
        logit_node, logit_edge, m_spec, logs_spec = self.network(z[k].to(self.device))
        x_node = Categorical(logits=logit_node).sample()
        x_edge = Categorical(logits=logit_edge).sample()
        x_spec = Normal(m_spec, torch.nn.functional.softplus(logs_spec)+1e-8).sample()
        return x_node, x_edge, x_spec


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
                x_spec: torch.Tensor,
                z: Optional[torch.Tensor]=None,
                log_w: Optional[torch.Tensor]=None,
                seed: Optional[int]=None
                ):
        if z is None:
            z, log_w = self.sampler(seed=seed)
        log_prob = self.decoder(x_node, x_edge, x_spec, z, self.num_chunks)
        log_prob = torch.logsumexp(log_prob + log_w.to(self.device).unsqueeze(0), dim=1)
        return log_prob

    @torch.no_grad
    def sample(self, num_samples: int):
        z, log_w = self.sampler()
        k = Categorical(logits=log_w).sample([num_samples])
        x_node, x_edge, x_spec = self.decoder.sample(z, k)
        return x_node, x_edge, x_spec
    
    @torch.no_grad
    def sample_given_spectrum(self, x_spec: torch.Tensor):
        # num samples is give by the batch size of x_spec
        z, log_w = self.sampler()
        log_spec = self.decoder.forward_spec(x_spec, z, self.num_chunks)
        log_w = log_w.unsqueeze(0) + log_spec

        k = Categorical(logits=log_w).sample()
        x_node, x_edge, _ = self.decoder.sample(z, k)
        return x_node, x_edge
    
    @torch.no_grad
    def predict_spectrum(self, x_node, x_edge):
        # returns E[Y|G] and Var[Y|G], where Y is spectrum and G is graph
        z, log_w = self.sampler()
        log_graph = self.decoder.forward_graph(x_node, x_edge, z, self.num_chunks)
        log_w = log_w+log_graph  

        w = torch.softmax(log_w, dim=-1).unsqueeze(-1)  # (batch_size, n_comps)
        m_k, s_k = self.decoder.distr_spec(z, self.num_chunks)
        
        m = torch.sum(w * m_k, dim=1)
        s = torch.sum(w * (m_k**2 + s_k**2), dim=1) + m**2

        return m, s



class PGCFFNNSpec(nn.Module):
    def __init__(self,
                 nd_n: int,
                 nd_s: int,
                 nk_n: int,
                 nk_e: int,
                 nz: int,
                 nh: int,
                 nl_b: int,
                 nl_n: int,
                 nl_e: int,
                 nl_s: int,
                 nb: int,
                 nc: int,
                 device: Optional[str]='cuda'
                 ):
        super(PGCFFNNSpec, self).__init__()
        nd_e = nd_n * (nd_n - 1) // 2

        self.nd_node = nd_n
        self.nd_edge = nd_e
        self.nd_spec = nd_s
        self.nk_node = nk_n
        self.nk_edge = nk_e

        backbone = BackFFNN(nd_n, nd_e, nd_s, nk_n, nk_e, nz, nh, nl_b, nl_n, nl_e, nl_s)
        self.network = ContinuousMixture(
            decoder=HybridDecoder(backbone, device=device),
            sampler=GaussianSampler(nz, nb, device=device),
            num_chunks=nc,
            device=device
        )
        self.mask = torch.tril(torch.ones(self.nd_node, self.nd_node, dtype=torch.bool), diagonal=-1)

        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor, a: torch.Tensor, spec: torch.Tensor):
        return self.network(x, a[:, self.mask].view(-1, self.nd_edge), spec)

    def logpdf(self, x: torch.Tensor, a: torch.Tensor, spec: torch.Tensor):
        return self(x, a, spec).mean()

    def sample(self, num_samples: int):
        x, l, spec = self.network.sample(num_samples)
        x = x.cpu()
        l = l.cpu()

        a = torch.zeros(num_samples, self.nd_node, self.nd_node, dtype=torch.long)
        a[:, self.mask] = l
        a.transpose(1, 2)[:, self.mask] = l

        return x, a #, spec
    
    def sample_given_spectrum(self, spec: torch.Tensor):
        x, l = self.network.sample_given_spectrum(spec)
        x = x.cpu()
        l = l.cpu()

        a = torch.zeros(len(x), self.nd_node, self.nd_node, dtype=torch.long)
        a[:, self.mask] = l
        a.transpose(1, 2)[:, self.mask] = l

        return x, a #, spec
    
    def predict_spectrum(self, x: torch.Tensor, a: torch.Tensor):
        return self.network.predict_spectrum(x, a[:, self.mask].view(-1, self.nd_edge))

MODELS = {
    'pgc_ffnn_spec': PGCFFNNSpec
}


if __name__ == '__main__':
    import json


    with open(f'config/qm9/pgc_ffn_spec.json', 'r') as f:
        hyperpars = json.load(f)

    hps = hyperpars['model_hpars']
    model = PGCFFNNSpec(**hps)
    print(model)

    x, a = model.sample(64) 
    x, a = x.to('cuda'), a.to('cuda')
    m_spec, s_spec = model.predict_spectrum(x, a)
    m_spec = m_spec.to('cuda') # mean_spectra
    print(x.shape, a.shape, m_spec.shape)

    lkl = model.forward(x, a, m_spec)   
    print(lkl.shape)
import math
import torch
import torch.nn as nn
import qmcpy
import models.moflow as mf

from torch.distributions import Categorical
from typing import Callable, Optional, List
from models.spn_utils import ohe2cat, cat2ohe

# the continuous mixture model is based on the following implementation: https://github.com/AlCorreia/cm-tpm

def ffnn_decoder(ni: int,
                 no: int,
                 nl: int,
                 batch_norm: bool,
                 final_act: Optional[any]=None
                 ):
        if ni != no:
            nh = torch.arange(ni, no, (no - ni) / nl, dtype=torch.int)
        else:
            nh = torch.full((nl, ), ni)
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

def conv_decoder(nz: int,
                 ns: int,
                 nc: int,
                 arch: int,
                 ):
    if arch == 0:
        decoder = nn.Sequential(
                # (n,   nz,  1,  1)
                nn.ConvTranspose2d(nz,   ns*8, 3, 1, 0, bias=False), # (n, ns*8,  3,  3)
                nn.BatchNorm2d(ns*8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ns*8, ns*4, 3, 2, 1, bias=False), # (n, ns*4,  5,  5)
                nn.BatchNorm2d(ns*4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ns*4,   nc, 3, 2, 1, bias=False), # (n,   nc,  9,  9)
            )
    else:
        decoder = nn.Sequential(
                # (n,   nz,  1,  1)
                nn.ConvTranspose2d(nz,   ns*8, 4, 3, 0, bias=False), # (n, ns*8,  3,  3)
                nn.BatchNorm2d(ns*8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ns*8, ns*4, 4, 3, 0, bias=False), # (n, ns*4,  5,  5)
                nn.BatchNorm2d(ns*4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ns*4,   nc, 4, 3, 1, bias=False), # (n,   nc,  9,  9)
            )
        # decoder = nn.Sequential(
        #         # (n,   nz,  1,  1)
        #         nn.ConvTranspose2d(nz,   ns*8, 3, 1, 0, bias=False), # (n, ns*8,  3,  3)
        #         nn.BatchNorm2d(ns*8),
        #         nn.ReLU(True),
        #         nn.ConvTranspose2d(ns*8, ns*6, 3, 2, 1, bias=False), # (n, ns*6,  5,  5)
        #         nn.BatchNorm2d(ns*6),
        #         nn.ReLU(True),
        #         nn.ConvTranspose2d(ns*6, ns*4, 3, 2, 1, bias=False), # (n, ns*4,  9, 9)
        #         nn.BatchNorm2d(ns*4),
        #         nn.ReLU(True),
        #         nn.ConvTranspose2d(ns*4, ns*2, 3, 2, 1, bias=False), # (n, ns*2, 17, 17)
        #         nn.BatchNorm2d(ns*2),
        #         nn.ReLU(True),
        #         nn.ConvTranspose2d(ns*2,   ns, 3, 2, 1, bias=False), # (n,   ns, 33, 33)
        #         nn.BatchNorm2d(ns),
        #         nn.ReLU(True),
        #         nn.ConvTranspose2d(  ns,   nc, 6, 1, 0, bias=False), # (n,   nc, 38, 38)
        #     )

    return decoder

class BackFFNN(nn.Module):
    def __init__(self,
                 nd_node: int,
                 nd_edge: int,
                 nk_node: int,
                 nk_edge: int,
                 nz_back: int,
                 nh_back: int,
                 nl_back: int,
                 nl_node: int,
                 nl_edge: int,
                 device: Optional[str]='cuda'
                 ):
        super(BackFFNN, self).__init__()

        self.nd_node = nd_node
        self.nd_edge = nd_edge
        self.nk_node = nk_node
        self.nk_edge = nk_edge
        self.net_back = ffnn_decoder(nz_back,            nh_back,  nl_back, True )
        self.net_node = ffnn_decoder(nh_back//2, nd_node*nk_node,  nl_node, False)
        self.net_edge = ffnn_decoder(nh_back//2, nd_edge*nk_edge,  nl_edge, False)
        self.device = device

    def forward(self, z):                                    # (chunk_size, nz_back)
        h_back = self.net_back(z)                            # (chunk_size, nh_back)
        h_node, h_edge = torch.chunk(h_back, 2, 1)           # (chunk_size, nh_back/2), (chunk_size, nh_back/2)
        h_node = self.net_node(h_node)                       # (chunk_size, nd_node*nk_node)
        h_edge = self.net_edge(h_edge)                       # (chunk_size, nd_edge*nk_edge)
        h_node = h_node.view(-1, self.nd_node, self.nk_node) # (chunk_size, nd_node, nk_node)
        h_edge = h_edge.view(-1, self.nd_edge, self.nk_edge) # (chunk_size, nd_edge, nk_edge)
        return h_node, h_edge

class BackConv(nn.Module):
    def __init__(self,
                 nd_node: int,
                 nd_edge: int,
                 nk_node: int,
                 nk_edge: int,
                 nz_back: int,
                 nh_back: int,
                 nl_back: int,
                 nl_node: int,
                 ns_edge: int,
                 arch: int,
                 device: Optional[str]='cuda'
                 ):
        super(BackConv, self).__init__()

        self.nd_node = nd_node
        self.nd_edge = nd_edge
        self.nk_node = nk_node
        self.nk_edge = nk_edge
        self.net_back = ffnn_decoder(nz_back, nh_back, nl_back, True )
        self.net_node = ffnn_decoder(nd_node*nk_node, nd_node*nk_node, nl_node, False)
        self.net_edge = conv_decoder(nd_edge*nk_edge, ns_edge, nk_edge, arch)
        self.device = device

    def forward(self, z):                                    # (chunk_size, nz_back)
        h_back = self.net_back(z)                            # (chunk_size, nh_back)
        # h_node, h_edge = torch.chunk(h_back, 2, 1)           # (chunk_size, nh_back/2), (chunk_size, nh_back/2)
        h_node, h_edge = h_back[:, :self.nd_node*self.nk_node], h_back[:, self.nd_node*self.nk_node:]
        h_edge = h_edge.unsqueeze(2).unsqueeze(3)            # (chunk_size, nh_back/2, 1, 1)
        h_node = self.net_node(h_node)                       # (chunk_size, nd_node*nk_node)
        h_edge = self.net_edge(h_edge)                       # (chunk_size, nk_edge, nd_node, nd_node)
        h_edge = torch.movedim(h_edge, 1, -1)                # (chunk_size, nd_node, nd_node, nk_edge)
        h_node = h_node.view(-1, self.nd_node, self.nk_node) # (chunk_size, nd_node, nk_node)
        h_edge = h_edge.view(-1, self.nd_edge, self.nk_edge) # (chunk_size, nd_edge, nk_edge)
        return h_node, h_edge

class BackFlow(nn.Module):
    def __init__(self,
                 # For atom
                 nd_node: int,
                 nk_node: int,
                 hgnn_node: List[int],
                 hlin_node: List[int],
                 nf_node: int,
                 nb_node: int,
                 mask_row_size_list: List[int],
                 mask_row_stride_list: List[int],
                 af_node: bool,
                 # For bond
                 nk_edge: int,
                 nf_edge: int,
                 nb_edge: int,
                 sq_edge: int,
                 hcha_edge: List[int],
                 af_edge: bool,
                 lu_edge: int,
                 # General
                 learn_dist: bool,
                 device: Optional[str]='cuda'
                 ):
        """
        :param nd_node:                 Maximum number of atoms in a molecule
        :param nk_node:                 Number of atom types
        :param hgnn_node:               Hidden dimension list for graph convolution for atoms matrix, delimited list input
        :param hlin_node:               Hidden dimension list for linear transformation for atoms, delimited list input
        :param nf_node:                 Number of masked flow coupling layers per block for atom matrix
        :param nb_node:                 Number of flow blocks for atom matrix
        :param mask_row_size_list:      Mask row list for atom matrix, delimited list input
        :param mask_row_stride_list:    Mask row stride  list for atom matrix, delimited list input
        :param af_node:                 Using affine coupling layers for atom conditional graph flow
        :param nk_edge:                 Number of bond types
        :param nf_edge:                 Number of masked glow coupling layers per block for bond tensor
        :param nb_edge:                 Number of glow blocks for bond tensor
        :param sq_edge:                 Squeeze divisor, 3 for qm9, 2 for zinc250k
        :param hcha_edge:               Hidden channel list for bonds tensor, delimited list input
        :param af_edge:                 Using affine coupling layers for bonds glow
        :param lu_edge:                 Using L-U decomposition trick for 1-1 conv in bonds glow
        :param learn_dist:              Learn the distribution of feature matrix
        """
        super(BackFlow, self).__init__()
        self.nd_node = nd_node
        self.nk_node = nk_node
        self.nk_edge = nk_edge
        if learn_dist:
            self.ln_var = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('ln_var', torch.zeros(1))

        self.bond_model = mf.Glow(
            in_channel=nk_edge,       # 4
            n_flow=nf_edge,           # 10
            n_block=nb_edge,          # 1
            squeeze_fold=sq_edge,     # 3
            hidden_channel=hcha_edge, # [128, 128]
            affine=af_edge,           # True
            conv_lu=lu_edge           # 0,1,2
        )

        self.atom_model = mf.GlowOnGraph(
            n_node=nd_node,                                          # 9,
            in_dim=nk_node,                                          # 5,
            hidden_dim_dict={'gnn': hgnn_node, 'linear': hlin_node}, # {'gnn': [64], 'linear': [128, 64]},
            n_flow=nf_node,                                          # 27,
            n_block=nb_node,                                         # 1,
            mask_row_size_list=mask_row_size_list,                   # [1],
            mask_row_stride_list=mask_row_stride_list,               # [1],
            affine=af_node                                           # True
        )

        self.device = device
        self.to(device)

    def forward(self, z, true_a=None):
        batch_size = z.shape[0] # 100,  z.shape: (100,369)

        ns_node = self.nd_node * self.nk_node
        zx = z[:, :ns_node] # (100, 45)
        za = z[:, ns_node:] # (100, 324)

        if true_a is None:
            ha = za.reshape(batch_size, self.nk_edge, self.nd_node, self.nd_node) # (100,4,9,9)
            # ha = self.bond_model(ha)[0]
            ha = self.bond_model.reverse(ha)
            ha = ha + ha.permute(0, 1, 3, 2)
            ha = ha / 2
            a = ha.softmax(dim=1) # (100,4,9,9)
            max_bond = a.max(dim=1).values.reshape(batch_size, -1, self.nd_node, self.nd_node) # (100,1,9,9)
            a = torch.floor(a / max_bond) # (100,4,9,9) /  (100,1,9,9) --> (100,4,9,9)
        else:
            a = true_a

        hx = zx.reshape(batch_size, self.nd_node, self.nk_node)
        a_norm = mf.rescale_adj(a).to(hx)
        # x = self.atom_model(a_norm, hx)[0]
        x = self.atom_model.reverse(a_norm, hx)

        a = torch.movedim(a, 1, -1) # (100,4,9,9) --> (100,9,9,4)
        a = a.view(-1, self.nd_node*self.nd_node, self.nk_edge)

        return x, a # (100,9,5), (100,81,4)

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
                 network: nn.Module,
                 device: Optional[str]='cuda'
                 ):
        super(CategoricalDecoder, self).__init__()
        self.network = network
        self.device = device

    def forward(self,
                x_node: torch.Tensor,                                       # (batch_size, nd_node)
                x_edge: torch.Tensor,                                       # (batch_size, nd_edge)
                z: torch.Tensor,
                num_chunks: Optional[int]=None
                ):
        x_node = x_node.unsqueeze(1).float()                                # (batch_size, 1, nd_node)
        x_edge = x_edge.unsqueeze(1).float()                                # (batch_size, 1, nd_edge)

        log_prob = torch.zeros(len(x_node), len(z), device=self.device)     # (batch_size, num_chunks*chunk_size)
        for c in torch.arange(len(z)).chunk(num_chunks):
            logit_node, logit_edge = self.network(z[c, :].to(self.device))  # (chunk_size, nd_node, nk_node), (chunk_size, nd_edge, nk_edge)
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node) # (batch_size, chunk_size, nd_node)
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge) # (batch_size, chunk_size, nd_edge)
            log_prob[:, c] = log_prob_node.sum(dim=2) + log_prob_edge.sum(dim=2)

        return log_prob

    def sample(self, z: torch.Tensor, k: torch.Tensor):
        logit_node, logit_edge = self.network(z[k].to(self.device))
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


class MolSPNFFNNSort(nn.Module):
    def __init__(self,
                 nd_n: int,
                 nk_n: int,
                 nk_e: int,
                 nz: int,
                 nh: int,
                 nl_b: int,
                 nl_n: int,
                 nl_e: int,
                 nb: int,
                 nc: int,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNFFNNSort, self).__init__()
        nd_e = nd_n * (nd_n - 1) // 2

        self.nd_node = nd_n
        self.nd_edge = nd_e
        self.nk_node = nk_n
        self.nk_edge = nk_e

        backbone = BackFFNN(nd_n, nd_e, nk_n, nk_e, nz, nh, nl_b, nl_n, nl_e)
        self.network = ContinuousMixture(
            decoder=CategoricalDecoder(backbone, device=device),
            sampler=GaussianSampler(nz, nb, device=device),
            num_chunks=nc,
            device=device
        )
        self.mask = torch.tril(torch.ones(self.nd_node, self.nd_node, dtype=torch.bool), diagonal=-1)

        self.device = device
        self.to(device)

    def forward(self, x_ohe, a_ohe):
        x = x_ohe.to(self.device)
        a = a_ohe.to(self.device)
        x, a = ohe2cat(x, a)
        return self.network(x, a[:, self.mask].view(-1, self.nd_edge))

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        x, l = self.network.sample(num_samples)
        x = x.cpu()
        l = l.cpu()

        a = torch.zeros(num_samples, self.nd_node, self.nd_node, dtype=torch.long)
        a[self.mask.expand(a.size())] = l.view(num_samples*self.nd_edge)

        return cat2ohe(x, a, self.nk_node, self.nk_edge)

class MolSPNConvSort(nn.Module):
    def __init__(self,
                 nd_n: int,
                 nk_n: int,
                 nk_e: int,
                 nl_b: int,
                 nb: int,
                 nc: int,
                 nl_n: int,
                 ns_e: int,
                 arch: int,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNConvSort, self).__init__()
        nd_e = nd_n ** 2

        self.nd_node = nd_n
        self.nd_edge = nd_e
        self.nk_node = nk_n
        self.nk_edge = nk_e

        nz = nd_n*nk_n + nd_e*nk_e
        nh = nz

        backbone = BackConv(nd_n, nd_e, nk_n, nk_e, nz, nh, nl_b, nl_n, ns_e, arch)
        self.network = ContinuousMixture(
            decoder=CategoricalDecoder(backbone, device=device),
            sampler=GaussianSampler(nz, nb, device=device),
            num_chunks=nc,
            device=device
        )

        self.device = device
        self.to(device)

    def forward(self, x_ohe, a_ohe):
        x = x_ohe.to(self.device)
        a = a_ohe.to(self.device)
        x, a = ohe2cat(x, a)
        return self.network(x, a.view(-1, self.nd_edge))

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        x, a = self.network.sample(num_samples)
        x = x.cpu()
        a = a.cpu()
        return cat2ohe(x, a.view(-1, self.nd_node, self.nd_node), self.nk_node, self.nk_edge)


class MolSPNFlowSort(nn.Module):
    def __init__(self,
                 # Atoms
                 nd_n: int,
                 nk_n: int,
                 hgnn_n: List[int],
                 hlin_n: List[int],
                 nf_n: int,
                 nb_n: int,
                 mask_row_size_list: List[int],
                 mask_row_stride_list: List[int],
                 af_n: bool,
                 # Bonds
                 nk_e: int,
                 nf_e: int,
                 nb_e: int,
                 sq_e: int,
                 hcha_e: List[int],
                 af_e: bool,
                 lu_e: int,
                 # General
                 nb: int,
                 nc: int,
                 learn_dist: Optional[bool]=True,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNFlowSort, self).__init__()
        self.nd_node = nd_n
        self.nd_edge = nd_n**2
        self.nk_node = nk_n
        self.nk_edge = nk_e

        backbone = BackFlow(
            nd_n, nk_n, hgnn_n, hlin_n, nf_n, nb_n, mask_row_size_list, mask_row_stride_list, af_n,
            nk_e, nf_e, nb_e, sq_e, hcha_e, af_e, lu_e,
            learn_dist, device
            )
        self.network = ContinuousMixture(
            decoder=CategoricalDecoder(backbone, device=device),
            sampler=GaussianQMCSampler(nd_n*nk_n + nd_n**2*nk_e, nb, device=device),
            num_chunks=nc,
            device=device
        )

        self.device = device
        self.to(device)

    def forward(self, x_ohe, a_ohe):
        x = x_ohe.to(self.device)
        a = a_ohe.to(self.device)
        x, a = ohe2cat(x, a)
        return self.network(x, a.view(-1, self.nd_edge))

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        x, a = self.network.sample(num_samples)
        x = x.cpu()
        a = a.cpu()
        return cat2ohe(x, a.view(-1, self.nd_node, self.nd_node), self.nk_node, self.nk_edge)

MODELS = {
    'molspn_ffnn_sort': MolSPNFFNNSort,
    'molspn_conv_sort': MolSPNConvSort,
    'molspn_flow_sort': MolSPNFlowSort,
}

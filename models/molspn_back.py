import math
import torch
import torch.nn as nn

from torch.distributions import Categorical
from typing import Callable, Optional, List
from models.utils import ohe2cat, cat2ohe, BackFFNN, BackConv, BackFlow
from models.graph_transformer import GraphTransformer


class GaussianSampler:
    def __init__(self,
                 nd_node_z: int,
                 nk_node_z: int,
                 nk_edge_z: int,
                 num_samples: int,
                 sd_node: Optional[float]=10.0,
                 sd_edge: Optional[float]=10.0,
                 trainable_weights: Optional[bool]=False,
                 device: Optional[str]='cuda'
                 ):
        self.nd_node_z = nd_node_z
        self.nk_node_z = nk_node_z
        self.nk_edge_z = nk_edge_z

        self.sd_node = sd_node
        self.sd_edge = sd_edge
        self.num_samples = num_samples
        self.device = device

        self.w = torch.full((num_samples,), math.log(1 / num_samples), device=self.device, requires_grad=trainable_weights)
        self.m = torch.tril(torch.ones(nd_node_z, nd_node_z, dtype=torch.bool), diagonal=-1)

    def __call__(self):
        z_node = self.sd_node*torch.randn(self.num_samples, self.nd_node_z, self.nk_node_z, device=self.device)
        z_edge = torch.zeros(self.num_samples, self.nd_node_z, self.nd_node_z, self.nk_edge_z, device=self.device)

        d_edge = self.nd_node_z*(self.nd_node_z - 1)//2
        v_edge = self.sd_edge*torch.randn(self.num_samples*d_edge*self.nk_edge_z, device=self.device)
        z_edge[:, self.m,   :] = v_edge.view(-1, d_edge, self.nk_edge_z)
        z_edge[:, self.m.T, :] = v_edge.view(-1, d_edge, self.nk_edge_z)

        return z_node, z_edge, self.w

class CategoricalSampler:
    def __init__(self,
                 nd_node_z: int,
                 nk_node_z: int,
                 nk_edge_z: int,
                 num_samples: int,
                 trainable_weights: Optional[bool]=False,
                 device: Optional[str]='cuda'
                 ):
        self.nd_node_z = nd_node_z
        self.nk_node_z = nk_node_z
        self.nk_edge_z = nk_edge_z

        self.num_samples = num_samples
        self.device = device

        self.w = torch.full((num_samples,), math.log(1 / num_samples), device=self.device, requires_grad=trainable_weights)
        self.m = torch.tril(torch.ones(nd_node_z, nd_node_z, dtype=torch.bool), diagonal=-1)

    def __call__(self):
        d_edge = self.nd_node_z*(self.nd_node_z - 1)//2
        logit_node = torch.ones(self.nk_node_z, device=self.device)
        logit_edge = torch.ones(self.nk_edge_z, device=self.device)
        z_node = Categorical(logits=logit_node).sample((self.num_samples, self.nd_node_z)).float()
        v_edge = Categorical(logits=logit_edge).sample((self.num_samples*d_edge, )).float()

        z_edge = torch.zeros(self.num_samples, self.nd_node_z, self.nd_node_z, device=self.device)
        z_edge[:, self.m  ] = v_edge.view(-1, d_edge)
        z_edge[:, self.m.T] = v_edge.view(-1, d_edge)
        z_node, z_edge = cat2ohe(z_node, z_edge, self.nk_node_z, self.nk_edge_z)

        z_edge = torch.movedim(z_edge, 1, -1) # get rid of this in cat2ohe and adapt the flow model accordingly

        return z_node.float(), z_edge.float(), self.w


class CategoricalDecoder(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 num_chunks: Optional[int]=2,
                 device: Optional[str]='cuda'
                 ):
        super(CategoricalDecoder, self).__init__()
        self.network = network
        self.num_chunks = num_chunks
        self.device = device

    def forward(self,
                x_node: torch.Tensor,                                            # (batch_size, nd_node_x, nk_node_x)
                x_edge: torch.Tensor,                                            # (batch_size, nk_edge_x, nd_node_x, nd_node_x)
                z_node: torch.Tensor,                                            # (batch_size, nd_node_z, nk_node_z)
                z_edge: torch.Tensor                                             # (batch_size, nd_node_z, nd_node_z, nk_edge_z)
                ):
        x_node, x_edge = ohe2cat(x_node, x_edge)                                 # (batch_size, nd_node_x), (batch_size, nd_node_x, nd_node_x)
        x_node = x_node.unsqueeze(1).float()                                     # (batch_size, 1, nd_node)
        x_edge = x_edge.unsqueeze(1).float()                                     # (batch_size, 1, nd_node_x, nd_node_x)

        log_prob = torch.zeros(len(x_node), len(z_node), device=self.device)     # (batch_size, num_chunks*chunk_size)
        for c in torch.arange(len(z_node)).chunk(self.num_chunks):
            logit_node, logit_edge = self.network(z_node[c, :], z_edge[c, :])    # (chunk_size, nd_node_x, nk_node_x), (chunk_size, nd_node_x, nd_node_x, nk_edge_x)
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node)      # (batch_size, chunk_size, nd_node_x)
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge)      # (batch_size, chunk_size, nd_node_x, nd_node_x)
            log_prob[:, c] = log_prob_node.sum(dim=2) + log_prob_edge.sum(dim=(2, 3))

        return log_prob

    def sample(self, z_node: torch.Tensor, z_edge: torch.Tensor, k: torch.Tensor):
        logit_node, logit_edge = self.network(z_node[k], z_edge[k])
        x_node = Categorical(logits=logit_node).sample()
        x_edge = Categorical(logits=logit_edge).sample()
        x_node, x_edge = cat2ohe(x_node, x_edge, self.network.nk_node_x, self.network.nk_edge_x)
        return x_node, x_edge


class ContinuousMixture(nn.Module):
    def __init__(self,
                 decoder: nn.Module,
                 sampler: Callable,
                 device: Optional[str]='cuda'
                 ):
        super(ContinuousMixture, self).__init__()
        self.decoder = decoder
        self.sampler = sampler
        self.device = device

    def forward(self, x_node: torch.Tensor, x_edge: torch.Tensor):
        z_node, z_edge, w = self.sampler()
        log_prob = self.decoder(x_node, x_edge, z_node, z_edge)
        log_prob = torch.logsumexp(log_prob + w.unsqueeze(0), dim=1)
        return log_prob

    def sample(self, num_samples: int):
        z_node, z_edge, w = self.sampler()
        k = Categorical(logits=w).sample([num_samples])
        x_node, x_edge = self.decoder.sample(z_node, z_edge, k)
        return x_node, x_edge


class MolSPNFFNNSort(nn.Module):
    def __init__(self,
                 nd_nx: int,
                 nk_nx: int,
                 nk_ex: int,
                 nd_nz: int,
                 nk_nz: int,
                 nk_ez: int,
                 nh_n: int,
                 nh_e: int,
                 nl_b: int,
                 nl_n: int,
                 nl_e: int,
                 nb: int,
                 nc: int,
                 tw: bool,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNFFNNSort, self).__init__()

        backbone = BackFFNN(nd_nz, nk_nz, nk_ez, nd_nx, nk_nx, nk_ex, nh_n, nh_e, nl_b, nl_n, nl_e, device)
        self.network = ContinuousMixture(
            CategoricalDecoder(backbone, nc, device),
            CategoricalSampler(nd_nz, nk_nz, nk_ez, nb, trainable_weights=tw, device=device),
            device
        )
        self.device = device
        self.to(device)

    def forward(self, x, a):
        return self.network(x.to(self.device), a.to(self.device))

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        return self.network.sample(num_samples)

class MolSPNConvSort(nn.Module):
    def __init__(self,
                 nd_nx: int,
                 nk_nx: int,
                 nk_ex: int,
                 nd_nz: int,
                 nk_nz: int,
                 nk_ez: int,
                 nh_n: int,
                 nl_b: int,
                 nl_n: int,
                 ns_e: int,
                 nb: int,
                 nc: int,
                 tw: bool,
                 arch: int,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNConvSort, self).__init__()

        backbone = BackConv(nd_nz, nk_nz, nk_ez, nd_nx, nk_nx, nk_ex, nh_n, nl_b, nl_n, ns_e, arch, device)
        self.network = ContinuousMixture(
            CategoricalDecoder(backbone, nc, device),
            CategoricalSampler(nd_nz, nk_nz, nk_ez, nb, trainable_weights=tw, device=device),
            device
        )
        self.device = device
        self.to(device)

    def forward(self, x, a):
        return self.network(x.to(self.device), a.to(self.device))

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        return self.network.sample(num_samples)

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
            sampler=GaussianSampler(nd_n*nk_n + nd_n**2*nk_e, nb, device=device),
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

class MolSPNTranSort(nn.Module):
    def __init__(self,
                 nk_nx: int,
                 nk_ex: int,
                 nd_nz: int,
                 nk_nz: int,
                 nk_ez: int,
                 mh_n: int,
                 mh_e: int,
                 n_head: int,
                 nh_n: int,
                 nh_e: int,
                 df_n: int,
                 df_e: int,
                 nl: int,
                 nb: int,
                 nc: int,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNTranSort, self).__init__()

        backbone = GraphTransformer(nl, nk_nx, nk_ex, mh_n, mh_e, n_head, nh_n, nh_e, df_n, df_e)
        self.network = ContinuousMixture(
            CategoricalDecoder(backbone, nc, device),
            CategoricalSampler(nd_nz, nk_nz, nk_ez, nb, device=device),
            device
        )
        self.device = device
        self.to(device)

    def forward(self, x, a):
        return self.network(x.to(self.device), a.to(self.device))

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        return self.network.sample(num_samples)

MODELS = {
    'molspn_ffnn_sort': MolSPNFFNNSort,
    'molspn_conv_sort': MolSPNConvSort,
    'molspn_flow_sort': MolSPNFlowSort,
    'molspn_tran_sort': MolSPNTranSort,
}

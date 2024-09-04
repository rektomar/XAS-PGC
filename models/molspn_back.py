import torch
import torch.nn as nn

from torch.distributions import Categorical
from typing import Callable, Optional, List
from models.utils import ohe2cat, cat2ohe, DecoderFFNN, BackConv, BackFlow, CategoricalDecoder, CategoricalSampler
from models.graph_transformer import GraphTransformer


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
        x_node, x_edge = self.decoder.sample(z_node[k], z_edge[k])
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

        backbone = DecoderFFNN(nd_nz, nk_nz, nk_ez, nd_nx, nk_nx, nk_ex, nh_n, nh_e, nl_b, nl_n, nl_e, device)
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

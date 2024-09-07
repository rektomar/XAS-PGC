import math
import torch
import torch.nn as nn
import models.moflow as mf

from torch.distributions import Categorical, Normal
from typing import Optional, List


def cat2ohe(x, a, num_node_types, num_edge_types):
    x = torch.nn.functional.one_hot(x.to(torch.int64), num_classes=num_node_types)
    a = torch.nn.functional.one_hot(a.to(torch.int64), num_classes=num_edge_types)
    return x, a

def ohe2cat(x, a):
    x = torch.argmax(x, dim=-1)
    a = torch.argmax(a, dim=-1)
    return x, a

def zero_diagonal(a, device):
    bs, nd_node, _, nk_edge = a.shape
    mask = torch.eye(nd_node, dtype=bool, device=device)
    mask = mask.unsqueeze(0).unsqueeze(3).expand(bs, -1, -1, nk_edge)
    a[mask] = 0.
    return a

def ffnn_network(ni: int,
                 no: int,
                 nl: int,
                 batch_norm: bool,
                 final_act: Optional[any]=None
                 ):
        if ni != no:
            nh = torch.arange(ni, no, (no - ni) / nl, dtype=torch.int)
        else:
            nh = torch.full((nl, ), ni)
        network = nn.Sequential()
        for i in range(len(nh) - 1):
            network.append(nn.Linear(nh[i], nh[i + 1]))
            network.append(nn.ReLU())
            if batch_norm:
                # network.append(nn.BatchNorm1d(nh[i + 1]))
                network.append(nn.LayerNorm(nh[i + 1].item()))
        network.append(nn.Linear(nh[-1], no))
        if final_act is not None:
            network.append(final_act)
        return network

def conv_network(nz: int,
                 ns: int,
                 nc: int,
                 arch: int,
                 ):
    if arch == 0:
        network = nn.Sequential(
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
        network = nn.Sequential(
                # (n,   nz,  1,  1)
                nn.ConvTranspose2d(nz,   ns*8, 4, 3, 0, bias=False), # (n, ns*8,  3,  3)
                nn.BatchNorm2d(ns*8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ns*8, ns*4, 4, 3, 0, bias=False), # (n, ns*4,  5,  5)
                nn.BatchNorm2d(ns*4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ns*4,   nc, 4, 3, 1, bias=False), # (n,   nc,  9,  9)
            )
        # network = nn.Sequential(
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

    return network


class EncoderFFNN(nn.Module):
    def __init__(self,
                 nd_no: int,
                 nk_no: int,
                 nk_eo: int,
                 nd_ni: int,
                 nk_ni: int,
                 nk_ei: int,
                 nh_node: int,
                 nh_edge: int,
                 nl_back: int,
                 nl_node: int,
                 nl_edge: int,
                 device: Optional[str]='cuda'
                 ):
        super(EncoderFFNN, self).__init__()

        self.nd_no = nd_no
        self.nk_no = nk_no
        self.nk_eo = nk_eo

        self.nd_ni = nd_ni
        self.nk_ni = nk_ni
        self.nk_ei = nk_ei

        self.nh_node = nh_node
        self.nh_edge = nh_edge

        nz_node = nd_ni   *nk_ni
        nz_edge = nd_ni**2*nk_ei

        self.nz_node = nz_node

        self.net_node = ffnn_network(nd_no   *nk_no, nh_node,  nl_node, True, nn.ReLU())
        self.net_edge = ffnn_network(nd_no**2*nk_eo, nh_edge,  nl_edge, True, nn.ReLU())
        self.net_back = ffnn_network(nh_node+nh_edge, nz_node+nz_edge, nl_back, True)
        self.device = device

    def forward(self, x_node, x_edge):                                  # (bs, nd_no,  nk_no), (bs, nd_no, nd_no, nk_eo)
        x_node = x_node.reshape(-1, self.nd_no   *self.nk_no)           # (bs, nd_no  *nk_no)
        x_edge = x_edge.reshape(-1, self.nd_no**2*self.nk_eo)           # (bs, nd_no^2*nk_eo)
        h_node = self.net_node(x_node)                                  # (bs, nh_node)
        h_edge = self.net_edge(x_edge)                                  # (bs, nh_edge)
        h = torch.cat((h_node, h_edge), dim=1)                          # (bs, nh_node + nh_edge)
        h_back = self.net_back(h)                                       # (bs, nz_node + nz_edge)
        h_node = h_back[:, :self.nz_node]                               # (bs, nz_node)
        h_edge = h_back[:, self.nz_node:]                               # (bs, nz_edge)
        h_node = h_node.reshape(-1, self.nd_ni, self.nk_ni)             # (nz_mult*bs, nd_ni, nk_ni)
        h_edge = h_edge.reshape(-1, self.nd_ni, self.nd_ni, self.nk_ei) # (nz_mult*bs, nd_ni, nd_ni, nk_ei)
        h_edge = zero_diagonal(h_edge, self.device)
        h_edge = (h_edge + h_edge.transpose(1, 2)) / 2
        return h_node, h_edge

class DecoderFFNN(nn.Module):
    def __init__(self,
                 nd_ni: int,
                 nk_ni: int,
                 nk_ei: int,
                 nd_no: int,
                 nk_no: int,
                 nk_eo: int,
                 nh_node: int,
                 nh_edge: int,
                 nl_back: int,
                 nl_node: int,
                 nl_edge: int,
                 device: Optional[str]='cuda'
                 ):
        super(DecoderFFNN, self).__init__()

        self.nd_ni = nd_ni
        self.nk_ni = nk_ni
        self.nk_ei = nk_ei

        self.nd_no = nd_no
        self.nk_no = nk_no
        self.nk_eo = nk_eo

        self.nh_node = nh_node
        self.nh_edge = nh_edge

        nz_node = nd_ni   *nk_ni
        nz_edge = nd_ni**2*nk_ei

        self.net_back = ffnn_network(nz_node+nz_edge,        nh_node+nh_edge,  nl_back, True, nn.ReLU())
        self.net_node = ffnn_network(        nh_node, nd_no   *nk_no,  nl_node, True)
        self.net_edge = ffnn_network(        nh_edge, nd_no**2*nk_eo,  nl_edge, True)
        self.device = device

    def forward(self, z_node, z_edge):                               # (chunk_size, nd_ni,  nk_ni), (chunk_size, nd_ni, nd_ni, nk_ei)
        z_node = z_node.view(-1, self.nd_ni   *self.nk_ni)           # (chunk_size, nd_ni  *nk_ni)
        z_edge = z_edge.view(-1, self.nd_ni**2*self.nk_ei)           # (chunk_size, nd_ni^2*nk_ei)
        z = torch.cat((z_node, z_edge), dim=1)                       # (chunk_size, nd_ni*nk_ni + nd_ni^2*nk_ei)
        h_back = self.net_back(z)                                    # (chunk_size, nh_node + nh_edge)
        h_node = h_back[:, :self.nh_node]                            # (chunk_size, nh_node)
        h_edge = h_back[:, self.nh_node:]                            # (chunk_size, nh_edge)
        h_node = self.net_node(h_node)                               # (chunk_size, nd_no  *nk_no)
        h_edge = self.net_edge(h_edge)                               # (chunk_size, nd_no^2*nk_eo)
        h_node = h_node.view(-1, self.nd_no, self.nk_no)             # (chunk_size, nd_no, nk_no)
        h_edge = h_edge.view(-1, self.nd_no, self.nd_no, self.nk_eo) # (chunk_size, nd_no, nd_no, nk_eo)
        h_edge = zero_diagonal(h_edge, self.device)
        h_edge = (h_edge + h_edge.transpose(1, 2)) / 2
        return h_node, h_edge

class GraphXBlock(nn.Module):
    def __init__(self,
                 nk_ni: int,
                 nk_ei: int,
                 nk_no: int,
                 nk_eo: int,
                 nh_node: int,
                 nh_edge: int,
                 nl_node: int,
                 nl_edge: int,
                 nl_mm: int,
                 final_act: Optional[any]=None,
                 device: Optional[str]='cuda'
                 ):
        super(GraphXBlock, self).__init__()
        self.nk_ni = nk_ni
        self.nk_ei = nk_ei

        self.nk_no = nk_no
        self.nk_eo = nk_eo

        self.nh_node = nh_node
        self.nh_edge = nh_edge

        self.net_ni = ffnn_network(nk_ni, nh_node,  nl_node, True, nn.ReLU())
        self.net_ei = ffnn_network(nk_ei, nh_edge,  nl_edge, True, nn.ReLU())

        self.net_nm = ffnn_network(nh_edge, nh_node, nl_mm, True, nn.ReLU())
        self.net_em = ffnn_network(nh_node, nh_edge, nl_mm, True, nn.ReLU())

        self.lin_nm = nn.Linear(nh_node, nh_node)
        self.lin_em = nn.Linear(nh_edge, nh_edge)

        self.net_no = ffnn_network(nh_node, nk_no,  nl_node, True, final_act)
        self.net_eo = ffnn_network(nh_edge, nk_eo,  nl_edge, True, final_act)
        self.device = device

    def forward(self, x_node, x_edge):                      # (bs, nd_ni, nk_ni), (bs, nd_ni, nd_ni, nk_ei)
        nd_ni = x_node.shape[1]
        h_node = self.net_ni(x_node)                        # (bs, nd_ni, nh_node)
        h_edge = self.net_ei(x_edge)                        # (bs, nd_ni, nd_ni, nh_edge)

        h_node = h_node.sum(dim=1)                          # (bs, nh_node)
        h_edge = h_edge.sum(dim=(1,2))                      # (bs, nh_edge)
        h_node = self.lin_nm(h_node) + self.net_nm(h_edge)  # (bs, nh_node)
        h_edge = self.lin_em(h_edge) + self.net_em(h_node)  # (bs, nh_edge)

        h_node = h_node.unsqueeze(1).expand(-1, nd_ni, -1)                      # (bs, nd_ni, nk_ni)
        h_edge = h_edge.unsqueeze(1).unsqueeze(2).expand(-1, nd_ni, nd_ni, -1)  # (bs, nd_ni, nd_ni, nk_ei)

        h_node = self.net_no(h_node)                        # (bs, nd_ni, nk_no)
        h_edge = self.net_eo(h_edge)                        # (bs, nd_ni, nd_ni, nk_eo)
        return h_node, h_edge

class GraphXCoder(nn.Module):
    def __init__(self,
                 nk_ni: int,
                 nk_ei: int,
                 nk_no: int,
                 nk_eo: int,
                 nh_n: int,
                 nh_e: int,
                 nl_n: int,
                 nl_e: int,
                 nl_m: int,
                 nb: int,
                 device: Optional[str]='cuda'
                 ):
        super(GraphXCoder, self).__init__()
        self.nk_ni = nk_ni
        self.nk_ei = nk_ei

        self.nk_no = nk_no
        self.nk_eo = nk_eo

        self.blocks = nn.ModuleList()
        self.blocks.append(GraphXBlock(nk_ni, nk_ei, nh_n, nh_e, nh_n, nh_e, nl_n, nl_e, nl_m, device=device))
        for i in range(nb):
            self.blocks.append(GraphXBlock(nh_n, nh_e, nh_n, nh_e, nh_n, nh_e, nl_n, nl_e, nl_m, device=device))
            # if batch_norm:
            #     self.blocks.append(nn.LayerNorm(nh[i + 1].item()))
        self.blocks.append(GraphXBlock(nh_n, nh_e, nk_no, nk_eo, nh_n, nh_e, nl_n, nl_e, nl_m, device=device))

        self.device = device

    def forward(self, x_node, x_edge):
        h_node, h_edge = x_node, x_edge # (bs, nd_ni, nk_ni), (bs, nd_ni, nd_ni, nk_ei)
        for block in self.blocks:
            h_node, h_edge = block(h_node, h_edge)
        h_edge = zero_diagonal(h_edge, self.device)
        h_edge = (h_edge + h_edge.transpose(1, 2)) / 2
        return h_node, h_edge


class BackConv(nn.Module):
    def __init__(self,
                 nd_ni: int,
                 nk_ni: int,
                 nk_ei: int,
                 nd_no: int,
                 nk_no: int,
                 nk_eo: int,
                 nh_node: int,
                 nl_back: int,
                 nl_node: int,
                 ns_edge: int,
                 arch: int,
                 device: Optional[str]='cuda'
                 ):
        super(BackConv, self).__init__()

        self.nd_ni = nd_ni
        self.nk_ni = nk_ni
        self.nk_ei = nk_ei

        self.nd_no = nd_no
        self.nk_no = nk_no
        self.nk_eo = nk_eo

        self.nh_node = nh_node

        nz_node = nd_ni   *nk_ni
        nz_edge = nd_ni**2*nk_ei

        self.net_back = ffnn_network(nz_node+nz_edge,        nh_node+nz_edge,  nl_back, True, nn.ReLU())
        self.net_node = ffnn_network(        nh_node, nd_no   *nk_no,  nl_node, True)
        self.net_edge = conv_network(        nz_edge, ns_edge, nk_eo, arch)
        self.device = device

    def forward(self, z_node, z_edge):                     # (chunk_size, nd_ni,  nk_ni), (chunk_size, nd_ni, nd_ni, nk_ei)
        z_node = z_node.view(-1, self.nd_ni   *self.nk_ni) # (chunk_size, nd_ni  *nk_ni)
        z_edge = z_edge.view(-1, self.nd_ni**2*self.nk_ei) # (chunk_size, nd_ni^2*nk_ei)
        z = torch.cat((z_node, z_edge), dim=1)             # (chunk_size, nd_ni*nk_ni + nd_ni^2*nk_ei)
        h_back = self.net_back(z)                          # (chunk_size, nh_node + nz_edge)
        h_node = h_back[:, :self.nh_node]                  # (chunk_size, nh_node)
        h_edge = h_back[:, self.nh_node:]                  # (chunk_size, nz_edge)
        h_edge = h_edge.unsqueeze(2).unsqueeze(3)          # (chunk_size, nz_edge, 1, 1)
        h_node = self.net_node(h_node)                     # (chunk_size, nd_no*nk_no)
        h_edge = self.net_edge(h_edge)                     # (chunk_size, nk_eo, nd_no, nd_no)
        h_node = h_node.view(-1, self.nd_no, self.nk_no)   # (chunk_size, nd_no, nk_no)
        h_edge = torch.movedim(h_edge, 1, -1)              # (chunk_size, nd_no, nd_no, nk_eo)
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
                 nd_ni: int,
                 nk_ni: int,
                 nk_ei: int,
                 num_samples: Optional[int]=1,
                 sd_node: Optional[float]=1.0,
                 sd_edge: Optional[float]=1.0,
                 trainable_weights: Optional[bool]=False,
                 device: Optional[str]='cuda'
                 ):
        self.nd_ni = nd_ni
        self.nk_ni = nk_ni
        self.nk_ei = nk_ei

        self.sd_node = sd_node
        self.sd_edge = sd_edge
        self.num_samples = num_samples
        self.device = device

        self.w = torch.full((num_samples,), math.log(1 / num_samples), device=self.device, requires_grad=trainable_weights)
        self.m = torch.tril(torch.ones(nd_ni, nd_ni, dtype=torch.bool), diagonal=-1)

    def __call__(self, num_samples: Optional[any]=None):
        if num_samples == None:
            num_samples = self.num_samples

        z_node = self.sd_node*torch.randn(num_samples, self.nd_ni, self.nk_ni, device=self.device)
        z_edge = torch.zeros(num_samples, self.nd_ni, self.nd_ni, self.nk_ei, device=self.device)

        d_edge = self.nd_ni*(self.nd_ni - 1)//2
        v_edge = self.sd_edge*torch.randn(num_samples*d_edge*self.nk_ei, device=self.device)
        z_edge[:, self.m,   :] = v_edge.view(-1, d_edge, self.nk_ei)
        z_edge[:, self.m.T, :] = v_edge.view(-1, d_edge, self.nk_ei)

        return z_node, z_edge, self.w

class CategoricalSampler:
    def __init__(self,
                 nd_ni: int,
                 nk_ni: int,
                 nk_ei: int,
                 num_samples: Optional[int]=1,
                 trainable_weights: Optional[bool]=False,
                 device: Optional[str]='cuda'
                 ):
        self.nd_ni = nd_ni
        self.nk_ni = nk_ni
        self.nk_ei = nk_ei

        self.num_samples = num_samples
        self.device = device

        self.w = torch.full((num_samples,), math.log(1 / num_samples), device=self.device, requires_grad=trainable_weights)
        self.m = torch.tril(torch.ones(nd_ni, nd_ni, dtype=torch.bool), diagonal=-1)

    def __call__(self, num_samples: Optional[any]=None):
        if num_samples == None:
            num_samples = self.num_samples

        d_edge = self.nd_ni*(self.nd_ni - 1)//2
        logit_node = torch.ones(self.nk_ni, device=self.device)
        logit_edge = torch.ones(self.nk_ei, device=self.device)
        z_node = Categorical(logits=logit_node).sample((self.num_samples, self.nd_ni)).float()
        v_edge = Categorical(logits=logit_edge).sample((self.num_samples*d_edge, )).float()

        z_edge = torch.zeros(self.num_samples, self.nd_ni, self.nd_ni, device=self.device)
        z_edge[:, self.m  ] = v_edge.view(-1, d_edge)
        z_edge[:, self.m.T] = v_edge.view(-1, d_edge)
        z_node, z_edge = cat2ohe(z_node, z_edge, self.nk_ni, self.nk_ei)

        z_edge = torch.movedim(z_edge, 1, -1) # get rid of this in cat2ohe and adapt the flow model accordingly

        return z_node.float(), z_edge.float(), self.w


class GaussianEncoder(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 device: Optional[str]='cuda'
                 ):
        super(GaussianEncoder, self).__init__()
        self.network = network
        self.device = device

    def forward(self,
                x_node: torch.Tensor,                                                    # (batch_size, nd_no, nk_no)
                x_edge: torch.Tensor                                                     # (batch_size, nd_no, nd_no, nk_no)
                ):
        embed_node, embed_edge = self.network(x_node, x_edge)                            # (batch_size, nh_node_z, nk_ni), (batch_size, nh_edge_z, nh_edge_z, nk_ei)
        m_node, v_node = embed_node.chunk(2, dim=-1)                                     # (batch_size, nh_node_z, nk_ni), (batch_size, nh_node_z, nk_ni)
        m_edge, v_edge = embed_edge.chunk(2, dim=-1)                                     # (batch_size, nh_edge_z, nh_edge_z, nk_ei), (batch_size, nh_edge_z, nh_edge_z, nk_ei)

        kld_node = -0.5 * torch.sum(1 + v_node - m_node ** 2 - v_node.exp(), dim=2)      # (batch_size, nh_node_z)
        kld_edge = -0.5 * torch.sum(1 + v_edge - m_edge ** 2 - v_edge.exp(), dim=(2, 3)) # (batch_size, nh_edge_z)

        z_node = m_node + torch.exp(0.5*v_node)*torch.randn_like(v_node)                 # (batch_size, nh_node_z, nk_ni)
        z_edge = m_edge + torch.exp(0.5*v_edge)*torch.randn_like(v_edge)                 # (batch_size, nh_edge_z, nh_edge_z, nk_ni)
        z_edge = zero_diagonal(z_edge, self.device)
        z_edge = (z_edge + z_edge.transpose(1, 2)) / 2

        return kld_node + kld_edge, z_node, z_edge

    def sample(self, x_node: torch.Tensor, x_edge: torch.Tensor, num_samples: int):
        embed_node, embed_edge = self.network(x_node, x_edge)
        m_node, v_node = embed_node.chunk(2, dim=0)
        m_edge, v_edge = embed_edge.chunk(2, dim=0)

        z_node = Normal(m_node, torch.exp(0.5*v_node)).sample((num_samples,))
        z_edge = Normal(m_edge, torch.exp(0.5*v_edge)).sample((num_samples,))
        z_edge = zero_diagonal(z_edge, self.device)
        z_edge = (z_edge + z_edge.transpose(1, 2)) / 2

        return z_node, z_edge

class CategoricalDecoder(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 num_chunks: Optional[int]=1,
                 device: Optional[str]='cuda'
                 ):
        super(CategoricalDecoder, self).__init__()
        self.network = network
        self.num_chunks = num_chunks
        self.device = device

    def forward(self,
                x_node: torch.Tensor,                                            # (bs, nd_no, nk_no)
                x_edge: torch.Tensor,                                            # (bs, nd_no, nd_no, nk_eo)
                z_node: torch.Tensor,                                            # (bs, nd_ni, nk_ni)
                z_edge: torch.Tensor                                             # (bs, nd_ni, nd_ni, nk_ei)
                ):
        x_node, x_edge = ohe2cat(x_node, x_edge)                                 # (bs, nd_no), (bs, nd_no, nd_no)
        # x_node = x_node.unsqueeze(1).float()                                     # (bs, 1, nd_no)
        # x_edge = x_edge.unsqueeze(1).float()                                     # (bs, 1, nd_no, nd_no)
        x_node = x_node.float()                                                  # (bs, nd_no)
        x_edge = x_edge.float()                                                  # (bs, nd_no, nd_no)

        log_prob = torch.zeros(len(x_node), len(z_node), device=self.device)     # (bs, num_chunks*chunk_size)
        for c in torch.arange(len(z_node)).chunk(self.num_chunks):
            logit_node, logit_edge = self.network(z_node[c, :], z_edge[c, :])    # (chunk_size, nd_no, nk_no), (chunk_size, nd_no, nd_no, nk_eo)
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node)      # (bs, chunk_size, nd_no)
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge)      # (bs, chunk_size, nd_no, nd_no)
            # log_prob[:, c] = log_prob_node.sum(dim=2) + log_prob_edge.sum(dim=(2, 3))
            log_prob[:, c] = log_prob_node.sum(dim=1) + log_prob_edge.sum(dim=(1, 2))

        return log_prob

    def sample(self, z_node: torch.Tensor, z_edge: torch.Tensor):
        logit_node, logit_edge = self.network(z_node, z_edge)
        x_node = Categorical(logits=logit_node).sample()
        x_edge = Categorical(logits=logit_edge).sample()
        x_node, x_edge = cat2ohe(x_node, x_edge, self.network.nk_no, self.network.nk_eo)
        return x_node, x_edge

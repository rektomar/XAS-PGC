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
                 ni: int,
                 no: int,

                 n_xi: int,
                 n_ai: int,
                 n_yi: int,

                 n_xo: int,
                 n_ao: int,
                 n_yo: int,

                 h_x: int,
                 h_a: int,
                 h_y: int,

                 l_x: int,
                 l_a: int,
                 l_y: int,
                 l_b: int,

                 device: Optional[str]='cuda'
                 ):
        super(EncoderFFNN, self).__init__()

        self.ni = ni
        self.no = no

        self.n_xi = n_xi
        self.n_ai = n_ai
        self.n_yi = n_yi

        self.n_xo = n_xo
        self.n_ao = n_ao
        self.n_yo = n_yo

        self.h_x = h_x
        self.h_a = h_a
        self.h_y = h_y

        self.net_x = ffnn_network( ni   *n_xi,                     h_x, l_x, True, nn.ReLU())
        self.net_a = ffnn_network( ni**2*n_ai,                     h_a, l_a, True, nn.ReLU())
        self.net_y = ffnn_network(       n_yi,                     h_y, l_y, True, nn.ReLU())
        self.net_b = ffnn_network(h_x+h_a+h_y, no*n_xo+no**2*n_ao+n_yo, l_b, True)

        self.device = device

    def forward(self, x, a, y):                            # (bs, ni, n_xi), (bs, ni, ni, n_ai), (bs, n_yi)
        xx = x.reshape(-1, self.ni   *self.n_xi)           # (bs, ni*n_xi)
        aa = a.reshape(-1, self.ni**2*self.n_ai)           # (bs, ni^2*n_ai)

        hx = self.net_x(xx)                                # (bs, h_x)
        ha = self.net_a(aa)                                # (bs, h_a)
        hy = self.net_y(y)                                 # (bs, h_y)

        hh = torch.cat((hx, ha, hy), dim=1)                # (bs, h_x + h_a + h_y)
        hb = self.net_b(hh)                                # (bs, no*n_xo + no**2*n_ao + n_yo)

        dx = self.no*self.n_xo
        da = self.no**2*self.n_ao

        hx = hb[:, :dx]                                    # (bs, no*n_xo)
        ha = hb[:, dx:dx+da]                               # (bs, no**2*n_ao)
        hy = hb[:, dx+da:]                                 # (bs, n_yo)

        hx = hx.reshape(-1, self.no, self.n_xo)            # (bs, no, n_xo)
        ha = ha.reshape(-1, self.no, self.no, self.n_ao)   # (bs, no, no, n_ao)
        ha = zero_diagonal(ha, self.device)
        ha = (ha + ha.transpose(1, 2)) / 2

        return hx, ha, hy


class DecoderFFNN(nn.Module):
    def __init__(self,
                 ni: int,
                 no: int,

                 n_xi: int,
                 n_ai: int,
                 n_yi: int,

                 n_xo: int,
                 n_ao: int,
                 n_yo: int,

                 h_x: int,
                 h_a: int,
                 h_y: int,

                 l_x: int,
                 l_a: int,
                 l_y: int,
                 l_b: int,

                 device: Optional[str]='cuda'
                 ):
        super(DecoderFFNN, self).__init__()

        self.ni = ni
        self.no = no

        self.n_xi = n_xi
        self.n_ai = n_ai
        self.n_yi = n_yi

        self.n_xo = n_xo
        self.n_ao = n_ao
        self.n_yo = n_yo

        self.h_x = h_x
        self.h_a = h_a
        self.h_y = h_y

        self.net_b = ffnn_network(ni*n_xi+ni**2*n_ai+n_yi, h_x+h_a+h_a, l_b, True, nn.ReLU())
        self.net_x = ffnn_network(                    h_x,  no   *n_xo, l_x, True)
        self.net_a = ffnn_network(                    h_a,  no**2*n_ao, l_a, True)
        self.net_y = ffnn_network(                    h_a,        n_yo, l_y, True)

        self.device = device

    def forward(self, zx, za, zy):                    # (bs, ni, n_xi), (bs, ni, ni, n_ai), (bs, n_yi)
        zx = zx.view(-1, self.ni   *self.n_xi)        # (bs, ni*n_xi)
        za = za.view(-1, self.ni**2*self.n_ai)        # (bs, ni^2*n_ai

        zz = torch.cat((zx, za, zy), dim=1)           # (bs, ni*n_xi + ni^2*n_ai + n_yi)
        hb = self.net_b(zz)                           # (bs, h_x + h_a + h_y)

        hx = hb[:, :self.h_x]                         # (bs, h_x)
        ha = hb[:, self.h_x:self.h_x+self.h_a]        # (bs, h_a)
        hy = hb[:, self.h_x+self.h_a:]                # (bs, h_y)

        hx = self.net_x(hx)                           # (bs, no*n_xo)
        ha = self.net_a(ha)                           # (bs, no**2*n_ao)
        hy = self.net_y(hy)                           # (bs, n_yo)

        hx = hx.view(-1, self.no, self.n_xo)          # (bs, nd_no, nk_no)
        ha = ha.view(-1, self.no, self.no, self.n_ao) # (bs, nd_no, nd_no, nk_eo)
        ha = zero_diagonal(ha, self.device)
        ha = (ha + ha.transpose(1, 2)) / 2

        return hx, ha, hy


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
                 no: int,
                 n_xo: int,
                 n_ao: int,
                 n_yo: int,
                 num_samples: Optional[int]=1,
                 sd_x: Optional[float]=1.0,
                 sd_a: Optional[float]=1.0,
                 sd_y: Optional[float]=1.0,
                 trainable_weights: Optional[bool]=False,
                 device: Optional[str]='cuda'
                 ):
        self.no = no
        self.n_xo = n_xo
        self.n_ao = n_ao
        self.n_yo = n_yo

        self.sd_x = sd_x
        self.sd_a = sd_a
        self.sd_y = sd_y

        self.num_samples = num_samples
        self.device = device

        self.w = torch.full((num_samples,), math.log(1 / num_samples), device=self.device, requires_grad=trainable_weights)
        self.m = torch.tril(torch.ones(no, no, dtype=torch.bool), diagonal=-1)

    def __call__(self, num_samples: Optional[any]=None):
        if num_samples == None:
            num_samples = self.num_samples

        zx = self.sd_x*torch.randn(num_samples, self.no,          self.n_xo, device=self.device)
        zy = self.sd_y*torch.randn(num_samples,                   self.n_yo, device=self.device)
        za =           torch.zeros(num_samples, self.no, self.no, self.n_ao, device=self.device)

        d_ao = self.no*(self.no - 1)//2
        va = self.sd_a*torch.randn(num_samples*d_ao*self.n_ao, device=self.device)
        za[:, self.m,   :] = va.view(-1, d_ao, self.n_ao)
        za[:, self.m.T, :] = va.view(-1, d_ao, self.n_ao)

        return zx, za, zy, self.w


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
                x: torch.Tensor,                                             # (bs, ni, n_xi)
                a: torch.Tensor,                                             # (bs, ni, ni, n_ai)
                y: torch.Tensor                                              # (bs, n_yi)
                ):
        hx, ha, hy = self.network(x, a, y)                                   # (bs, no, n_xo), (bs, no, no, n_ao), (bs, n_yo)
        mx, vx = hx.chunk(2, dim=-1)                                         # (bs, no, n_xo/2), (bs, no, n_xo/2)
        ma, va = ha.chunk(2, dim=-1)                                         # (bs, no, no, n_ao/2), (bs, no, no, n_ao/2)
        my, vy = hy.chunk(2, dim=-1)                                         # (bs, n_yo/2), (bs, n_yo/2)

        kld_x = -0.5 * torch.sum(1 + vx - mx ** 2 - vx.exp(), dim=(1, 2))    # (bs)
        kld_a = -0.5 * torch.sum(1 + va - ma ** 2 - va.exp(), dim=(1, 2, 3)) # (bs)

        zx = mx + torch.exp(0.5*vx)*torch.randn_like(vx)                     # (bs, no, n_xo/2)
        za = ma + torch.exp(0.5*va)*torch.randn_like(va)                     # (bs, no, no, n_ao/2)
        zy = my + torch.exp(0.5*vy)*torch.randn_like(vy)                     # (bs, n_yo/2)

        za = zero_diagonal(za, self.device)
        za = (za + za.transpose(1, 2)) / 2

        return kld_x + kld_a, zx, za, zy

    def sample(self,
               x: torch.Tensor,                                           # (bs, ni, n_xi)
               a: torch.Tensor,                                           # (bs, ni, ni, n_ai)
               y: torch.Tensor,                                           # (bs, n_yi)
               num_samples: int
               ):
        hx, ha, hy = self.network(x, a, y)                                # (bs, no, n_xo), (bs, no, no, n_ao), (bs, n_yo)
        mx, vx = hx.chunk(2, dim=-1)                                      # (bs, no, n_xo/2), (bs, no, n_xo/2)
        ma, va = ha.chunk(2, dim=-1)                                      # (bs, no, no, n_ao/2), (bs, no, no, n_ao/2)
        my, vy = hy.chunk(2, dim=-1)                                      # (bs, n_yo/2), (bs, n_yo/2)

        zx = Normal(mx, torch.exp(0.5*vx)).sample((num_samples,))         # (bs, no, n_xo/2)
        za = Normal(ma, torch.exp(0.5*va)).sample((num_samples,))         # (bs, no, no, n_ao/2)
        zy = Normal(my, torch.exp(0.5*vy)).sample((num_samples,))         # (bs, n_yo/2)

        za = zero_diagonal(za, self.device)
        za = (za + za.transpose(1, 2)) / 2

        return zx, za, zy


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
                xx: torch.Tensor,                                                  # (bs, no, n_xo)
                xa: torch.Tensor,                                                  # (bs, no, no, n_ao)
                xy: torch.Tensor,                                                  # (bs, n_yo)
                zx: torch.Tensor,                                                  # (bs, ni, n_xi)
                za: torch.Tensor,                                                  # (bs, ni, ni, n_ai)
                zy: torch.Tensor                                                   # (bs, n_yi)
                ):
        xx, xa = ohe2cat(xx, xa)                                                   # (bs, ni), (bs, ni, ni)
        xx = xx.float()
        xa = xa.float()

        # log_prob = torch.zeros(len(xx), len(zx), device=self.device)               # (bs, num_chunks*chunk_size)
        # # print(log_prob.size())
        # for c in torch.arange(len(zx)).chunk(self.num_chunks):
        #     logit_x, logit_a, logit_y = self.network(zx[c, :], za[c, :], zy[c, :]) # (chunk_size, no, n_xo), (chunk_size, no, no, n_ao)
        #     log_prob_x = Categorical(logits=logit_x).log_prob(xx)                  # (bs, chunk_size, no)
        #     log_prob_a = Categorical(logits=logit_a).log_prob(xa)                  # (bs, chunk_size, no, no)
        #     log_prob[:, c] = log_prob_x.sum(dim=1) + log_prob_a.sum(dim=(1, 2))
        logit_x, logit_a, logit_y = self.network(zx, za, zy)
        log_prob_x = Categorical(logits=logit_x).log_prob(xx)
        log_prob_a = Categorical(logits=logit_a).log_prob(xa)
        log_prob = log_prob_x.sum(dim=1) + log_prob_a.sum(dim=(1, 2))
        return log_prob
    
    
    def cm_logpdf_marginal(self,
                x_node: torch.Tensor,                                            # (bs, nd_no, nk_no)
                x_edge: torch.Tensor,                                            # (bs, nd_no, nd_no, nk_eo)
                m_node: torch.Tensor,                                            # (bs, nd_no, nk_no)
                m_edge: torch.Tensor,                                            # (bs, nd_no, nd_no, nk_eo)
                z_node: torch.Tensor,                                            # (bs, nd_ni, nk_ni)
                z_edge: torch.Tensor,                                            # (bs, nd_ni, nd_ni, nk_ei)
                z_feat: torch.Tensor
                ):
        x_node, x_edge = ohe2cat(x_node, x_edge)                                 # (bs, nd_no), (bs, nd_no, nd_no)
        x_node = x_node.unsqueeze(1).float()                                     # (bs, 1, nd_no)
        x_edge = x_edge.unsqueeze(1).float()                                     # (bs, 1, nd_no, nd_no)
        m_node = m_node.unsqueeze(1)                                             # (bs, 1, nd_no)
        m_edge = m_edge.unsqueeze(1)                                             # (bs, 1, nd_no, nd_no)

        log_prob = torch.zeros(len(x_node), len(z_node), device=self.device)     # (bs, num_chunks*chunk_size)
        for c in torch.arange(len(z_node)).chunk(self.num_chunks):
            logit_node, logit_edge, _ = self.network(z_node[c, :], z_edge[c, :], z_feat[c, :])    # (chunk_size, nd_no, nk_no), (chunk_size, nd_no, nd_no, nk_eo)
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node)*m_node      # (bs, chunk_size, nd_no)
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge)*m_edge      # (bs, chunk_size, nd_no, nd_no)
            log_prob[:, c] = log_prob_node.sum(dim=-1) + log_prob_edge.sum(dim=(-2, -1))

        return log_prob

    def sample(self,
               zx: torch.Tensor,                                                   # (bs, ni, n_xi)
               za: torch.Tensor,                                                   # (bs, ni, ni, n_ai)
               zy: torch.Tensor                                                    # (bs, n_yi)
               ):
        logit_x, logit_a, logit_y = self.network(zx, za, zy)                       # (bs, no, n_xo), (bs, no, no, n_ao), (bs, n_yo)
        xx = Categorical(logits=logit_x).sample()                                  # (bs, no)
        xa = Categorical(logits=logit_a).sample()                                  # (bs, no, no)
        xx, xa = cat2ohe(xx, xa, self.network.n_xo, self.network.n_ao)             # (bs, no, n_xo), (bs, no, no, n_ao)
        return xx.cpu(), xa.cpu()

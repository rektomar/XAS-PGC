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
                network.append(nn.BatchNorm1d(nh[i + 1]))
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
                 nd_node_x: int,
                 nk_node_x: int,
                 nk_edge_x: int,
                 nd_node_z: int,
                 nk_node_z: int,
                 nk_edge_z: int,
                 nh_node: int,
                 nh_edge: int,
                 nl_back: int,
                 nl_node: int,
                 nl_edge: int,
                 nz_mult: Optional[int]=2,
                 device: Optional[str]='cuda'
                 ):
        super(EncoderFFNN, self).__init__()

        self.nd_node_x = nd_node_x
        self.nk_node_x = nk_node_x
        self.nk_edge_x = nk_edge_x

        self.nd_node_z = nd_node_z
        self.nk_node_z = nk_node_z
        self.nk_edge_z = nk_edge_z

        self.nh_node = nh_node
        self.nh_edge = nh_edge

        nz_node = nz_mult*nd_node_z   *nk_node_z
        nz_edge = nz_mult*nd_node_z**2*nk_edge_z

        self.nz_node = nz_node

        self.net_node = ffnn_network(nd_node_x   *nk_node_x, nh_node,  nl_node, True, nn.ReLU())
        self.net_edge = ffnn_network(nd_node_x**2*nk_edge_x, nh_edge,  nl_edge, True, nn.ReLU())
        self.net_back = ffnn_network(nh_node+nh_edge, nz_node+nz_edge, nl_back, True)
        self.device = device

    def forward(self, x_node, x_edge):                                              # (bs, nd_node_x,  nk_node_x), (bs, nd_node_x, nd_node_x, nk_edge_x)
        x_node = x_node.reshape(-1, self.nd_node_x   *self.nk_node_x)               # (bs, nd_node_x  *nk_node_x)
        x_edge = x_edge.reshape(-1, self.nd_node_x**2*self.nk_edge_x)               # (bs, nd_node_x^2*nk_edge_x)
        h_node = self.net_node(x_node)                                              # (bs, nh_node)
        h_edge = self.net_edge(x_edge)                                              # (bs, nh_edge)
        h = torch.cat((h_node, h_edge), dim=1)                                      # (bs, nh_node + nh_edge)
        h_back = self.net_back(h)                                                   # (bs, nz_node + nz_edge)
        h_node = h_back[:, :self.nz_node]                                           # (bs, nz_node)
        h_edge = h_back[:, self.nz_node:]                                           # (bs, nz_edge)
        h_node = h_node.reshape(-1, self.nd_node_z, self.nk_node_z)                 # (nz_mult*bs, nd_node_z, nk_node_z)
        h_edge = h_edge.reshape(-1, self.nd_node_z, self.nd_node_z, self.nk_edge_z) # (nz_mult*bs, nd_node_z, nd_node_z, nk_edge_z)
        h_edge = zero_diagonal(h_edge, self.device)
        h_edge = (h_edge + h_edge.transpose(1, 2)) / 2
        return h_node, h_edge

class DecoderFFNN(nn.Module):
    def __init__(self,
                 nd_node_z: int,
                 nk_node_z: int,
                 nk_edge_z: int,
                 nd_node_x: int,
                 nk_node_x: int,
                 nk_edge_x: int,
                 nh_node: int,
                 nh_edge: int,
                 nl_back: int,
                 nl_node: int,
                 nl_edge: int,
                 device: Optional[str]='cuda'
                 ):
        super(DecoderFFNN, self).__init__()

        self.nd_node_z = nd_node_z
        self.nk_node_z = nk_node_z
        self.nk_edge_z = nk_edge_z

        self.nd_node_x = nd_node_x
        self.nk_node_x = nk_node_x
        self.nk_edge_x = nk_edge_x

        self.nh_node = nh_node
        self.nh_edge = nh_edge

        nz_node = nd_node_z   *nk_node_z
        nz_edge = nd_node_z**2*nk_edge_z

        self.net_back = ffnn_network(nz_node+nz_edge,        nh_node+nh_edge,  nl_back, True, nn.ReLU())
        self.net_node = ffnn_network(        nh_node, nd_node_x   *nk_node_x,  nl_node, True)
        self.net_edge = ffnn_network(        nh_edge, nd_node_x**2*nk_edge_x,  nl_edge, True)
        self.device = device

    def forward(self, z_node, z_edge):                                           # (chunk_size, nd_node_z,  nk_node_z), (chunk_size, nd_node_z, nd_node_z, nk_edge_z)
        z_node = z_node.view(-1, self.nd_node_z   *self.nk_node_z)               # (chunk_size, nd_node_z  *nk_node_z)
        z_edge = z_edge.view(-1, self.nd_node_z**2*self.nk_edge_z)               # (chunk_size, nd_node_z^2*nk_edge_z)
        z = torch.cat((z_node, z_edge), dim=1)                                   # (chunk_size, nd_node_z*nk_node_z + nd_node_z^2*nk_edge_z)
        h_back = self.net_back(z)                                                # (chunk_size, nh_node + nh_edge)
        h_node = h_back[:, :self.nh_node]                                        # (chunk_size, nh_node)
        h_edge = h_back[:, self.nh_node:]                                        # (chunk_size, nh_edge)
        h_node = self.net_node(h_node)                                           # (chunk_size, nd_node_x  *nk_node_x)
        h_edge = self.net_edge(h_edge)                                           # (chunk_size, nd_node_x^2*nk_edge_x)
        h_node = h_node.view(-1, self.nd_node_x, self.nk_node_x)                 # (chunk_size, nd_node_x, nk_node_x)
        h_edge = h_edge.view(-1, self.nd_node_x, self.nd_node_x, self.nk_edge_x) # (chunk_size, nd_node_x, nd_node_x, nk_edge_x)
        h_edge = zero_diagonal(h_edge, self.device)
        h_edge = (h_edge + h_edge.transpose(1, 2)) / 2
        return h_node, h_edge



class BackConv(nn.Module):
    def __init__(self,
                 nd_node_z: int,
                 nk_node_z: int,
                 nk_edge_z: int,
                 nd_node_x: int,
                 nk_node_x: int,
                 nk_edge_x: int,
                 nh_node: int,
                 nl_back: int,
                 nl_node: int,
                 ns_edge: int,
                 arch: int,
                 device: Optional[str]='cuda'
                 ):
        super(BackConv, self).__init__()

        self.nd_node_z = nd_node_z
        self.nk_node_z = nk_node_z
        self.nk_edge_z = nk_edge_z

        self.nd_node_x = nd_node_x
        self.nk_node_x = nk_node_x
        self.nk_edge_x = nk_edge_x

        self.nh_node = nh_node

        nz_node = nd_node_z   *nk_node_z
        nz_edge = nd_node_z**2*nk_edge_z

        self.net_back = ffnn_network(nz_node+nz_edge,        nh_node+nz_edge,  nl_back, True, nn.ReLU())
        self.net_node = ffnn_network(        nh_node, nd_node_x   *nk_node_x,  nl_node, True)
        self.net_edge = conv_network(        nz_edge, ns_edge, nk_edge_x, arch)
        self.device = device

    def forward(self, z_node, z_edge):                             # (chunk_size, nd_node_z,  nk_node_z), (chunk_size, nd_node_z, nd_node_z, nk_edge_z)
        z_node = z_node.view(-1, self.nd_node_z   *self.nk_node_z) # (chunk_size, nd_node_z  *nk_node_z)
        z_edge = z_edge.view(-1, self.nd_node_z**2*self.nk_edge_z) # (chunk_size, nd_node_z^2*nk_edge_z)
        z = torch.cat((z_node, z_edge), dim=1)                     # (chunk_size, nd_node_z*nk_node_z + nd_node_z^2*nk_edge_z)
        h_back = self.net_back(z)                                  # (chunk_size, nh_node + nz_edge)
        h_node = h_back[:, :self.nh_node]                          # (chunk_size, nh_node)
        h_edge = h_back[:, self.nh_node:]                          # (chunk_size, nz_edge)
        h_edge = h_edge.unsqueeze(2).unsqueeze(3)                  # (chunk_size, nz_edge, 1, 1)
        h_node = self.net_node(h_node)                             # (chunk_size, nd_node_x*nk_node_x)
        h_edge = self.net_edge(h_edge)                             # (chunk_size, nk_edge_x, nd_node_x, nd_node_x)
        h_node = h_node.view(-1, self.nd_node_x, self.nk_node_x)   # (chunk_size, nd_node_x, nk_node_x)
        h_edge = torch.movedim(h_edge, 1, -1)                      # (chunk_size, nd_node_x, nd_node_x, nk_edge_x)
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
                 nd_node_z: int,
                 nk_node_z: int,
                 nk_edge_z: int,
                 num_samples: Optional[int]=1,
                 sd_node: Optional[float]=1.0,
                 sd_edge: Optional[float]=1.0,
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

    def __call__(self, num_samples: Optional[any]=None):
        if num_samples == None:
            num_samples = self.num_samples

        z_node = self.sd_node*torch.randn(num_samples, self.nd_node_z, self.nk_node_z, device=self.device)
        z_edge = torch.zeros(num_samples, self.nd_node_z, self.nd_node_z, self.nk_edge_z, device=self.device)

        d_edge = self.nd_node_z*(self.nd_node_z - 1)//2
        v_edge = self.sd_edge*torch.randn(num_samples*d_edge*self.nk_edge_z, device=self.device)
        z_edge[:, self.m,   :] = v_edge.view(-1, d_edge, self.nk_edge_z)
        z_edge[:, self.m.T, :] = v_edge.view(-1, d_edge, self.nk_edge_z)

        return z_node, z_edge, self.w

class CategoricalSampler:
    def __init__(self,
                 nd_node_z: int,
                 nk_node_z: int,
                 nk_edge_z: int,
                 num_samples: Optional[int]=1,
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

    def __call__(self, num_samples: Optional[any]=None):
        if num_samples == None:
            num_samples = self.num_samples

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


class GaussianEncoder(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 device: Optional[str]='cuda'
                 ):
        super(GaussianEncoder, self).__init__()
        self.network = network
        self.device = device

    def forward(self,
                x_node: torch.Tensor,                                                    # (batch_size, nd_node_x, nk_node_x)
                x_edge: torch.Tensor                                                     # (batch_size, nd_edge_x, nd_node_x, nk_node_x)
                ):
        embed_node, embed_edge = self.network(x_node, x_edge)                            # (2*batch_size, nh_node_z, nk_node_z), (2*batch_size, nh_edge_z, nh_edge_z, nk_edge_z)
        m_node, v_node = embed_node.chunk(2, dim=0)                                      # (batch_size, nh_node_z, nk_node_z), (batch_size, nh_node_z, nk_node_z)
        m_edge, v_edge = embed_edge.chunk(2, dim=0)                                      # (batch_size, nh_edge_z, nh_edge_z, nk_edge_z), (batch_size, nh_edge_z, nh_edge_z, nk_edge_z)

        kld_node = -0.5 * torch.sum(1 + v_node - m_node ** 2 - v_node.exp(), dim=2)      # (batch_size, nh_node_z)
        kld_edge = -0.5 * torch.sum(1 + v_edge - m_edge ** 2 - v_edge.exp(), dim=(2, 3)) # (batch_size, nh_edge_z)

        z_node = m_node + torch.exp(0.5*v_node)*torch.randn_like(v_node)                 # (batch_size, nh_node_z, nk_node_z)
        z_edge = m_edge + torch.exp(0.5*v_edge)*torch.randn_like(v_edge)                 # (batch_size, nh_edge_z, nh_edge_z, nk_node_z)
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
                x_node: torch.Tensor,                                            # (bs, nd_node_x, nk_node_x)
                x_edge: torch.Tensor,                                            # (bs, nd_node_x, nd_node_x, nk_edge_x)
                z_node: torch.Tensor,                                            # (bs, nd_node_z, nk_node_z)
                z_edge: torch.Tensor                                             # (bs, nd_node_z, nd_node_z, nk_edge_z)
                ):
        x_node, x_edge = ohe2cat(x_node, x_edge)                                 # (bs, nd_node_x), (bs, nd_node_x, nd_node_x)
        x_node = x_node.unsqueeze(1).float()                                     # (bs, 1, nd_node_x)
        x_edge = x_edge.unsqueeze(1).float()                                     # (bs, 1, nd_node_x, nd_node_x)

        log_prob = torch.zeros(len(x_node), len(z_node), device=self.device)     # (bs, num_chunks*chunk_size)
        for c in torch.arange(len(z_node)).chunk(self.num_chunks):
            logit_node, logit_edge = self.network(z_node[c, :], z_edge[c, :])    # (chunk_size, nd_node_x, nk_node_x), (chunk_size, nd_node_x, nd_node_x, nk_edge_x)
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node)      # (bs, chunk_size, nd_node_x)
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge)      # (bs, chunk_size, nd_node_x, nd_node_x)
            log_prob[:, c] = log_prob_node.sum(dim=2) + log_prob_edge.sum(dim=(2, 3))

        return log_prob

    def sample(self, z_node: torch.Tensor, z_edge: torch.Tensor):
        logit_node, logit_edge = self.network(z_node, z_edge)
        x_node = Categorical(logits=logit_node).sample()
        x_edge = Categorical(logits=logit_edge).sample()
        x_node, x_edge = cat2ohe(x_node, x_edge, self.network.nk_node_x, self.network.nk_edge_x)
        return x_node, x_edge

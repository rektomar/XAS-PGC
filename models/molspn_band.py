import math
import torch
import torch.nn as nn
import itertools

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from models.utils import ohe2cat, cat2ohe
from utils.graphs import unflatten
from tqdm import tqdm


def marginalize_nodes(network, nd_nodes, num_full):
    num_empty = nd_nodes - num_full
    with torch.no_grad():
        if num_empty > 0:
            mx = torch.zeros(nd_nodes, dtype=torch.bool)
            mx[num_full:] = True
            marginalization_idx = torch.arange(nd_nodes)[mx.view(-1)]

            network.set_marginalization_idx(marginalization_idx)
        else:
            network.set_marginalization_idx(None)

def marginalize_edges(network, nd_nodes, bw, num_full):
    num_empty = nd_nodes - num_full
    with torch.no_grad():
        if num_empty > 0:
            ma = torch.zeros(nd_nodes, bw, dtype=torch.bool)
            ma[num_full:, :] = True
            marginalization_idx = torch.arange(nd_nodes * bw)[ma.flatten()]

            network.set_marginalization_idx(marginalization_idx)
        else:
            network.set_marginalization_idx(None)


class MolSPNBandCore(nn.Module):
    def __init__(self,
                 nc,
                 nd_n,
                 nk_n,
                 nk_e,
                 nl_n,
                 nl_e,
                 nr_n,
                 nr_e,
                 ns_n,
                 ns_e,
                 ni_n,
                 ni_e,
                 bw,
                 device='cuda'):
        super().__init__()
        nd_e = nd_n * bw
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e
        self.bw = bw

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        args_nodes = EinsumNetwork.Args(
            num_var=nd_n,
            num_dims=1,
            num_input_distributions=ni_n,
            num_sums=ns_n,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk_n},
            use_em=False)
        args_edges = EinsumNetwork.Args(
            num_var=nd_e,
            num_dims=1,
            num_input_distributions=ni_e,
            num_sums=ns_e,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk_e},
            use_em=False)

        self.network_nodes = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
        self.network_edges = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
        self.network_nodes.initialize()
        self.network_edges.initialize()

        self.logits_n = nn.Parameter(torch.ones(nd_n,   device=device), requires_grad=True)
        self.logits_b = nn.Parameter(torch.ones(bw + 1, device=device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.ones(1, nc,  device=device), requires_grad=True)

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, xx, aa, num_full):
        pass

    def forward(self, x, a):
        m_nodes =  (x != self.nk_nodes-1)
        m_bands = ~(a == self.nk_edges-1).all(dim=1)
        n_nodes = m_nodes.sum(dim=1) - 1
        n_bands = m_bands.sum(dim=1)
        d_nodes = torch.distributions.Categorical(logits=self.logits_n)
        d_bands = torch.distributions.Categorical(logits=self.logits_b)

        self.network_nodes.set_marginalization_mask(m_nodes)
        self.network_edges.set_marginalization_mask(m_bands.unsqueeze(1).expand(-1, self.nd_nodes, -1).reshape(-1, self.nd_edges))

        return d_nodes.log_prob(n_nodes) + d_bands.log_prob(n_bands) + self._forward(x, a)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        # x = torch.zeros(num_samples, self.nd_nodes)
        # l = torch.zeros(num_samples, self.nd_nodes, self.bw)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, device=self.device)

        dist_n = torch.distributions.Categorical(logits=self.logits_n)
        dist_b = torch.distributions.Categorical(logits=self.logits_b)
        dist_w = torch.distributions.Categorical(logits=self.logits_w)

        samp_n = dist_n.sample((num_samples, ))
        samp_b = dist_b.sample((num_samples, ))
        samp_w = dist_w.sample((num_samples, )).squeeze()

        m_nodes = torch.arange(self.nd_nodes, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        m_bands = torch.arange(self.bw,       device=self.device).unsqueeze(0) <  samp_b.unsqueeze(1)
        m_bands = m_bands.unsqueeze(1).expand(-1, self.nd_nodes, -1).reshape(-1, self.nd_edges)

        self.network_nodes.set_marginalization_mask(m_nodes)
        self.network_edges.set_marginalization_mask(m_bands)
        x = self.network_nodes.sample(num_samples, class_idxs=samp_w)
        l = self.network_edges.sample(num_samples, class_idxs=samp_w)
        x[~m_nodes] = self.nk_nodes - 1
        l[~m_bands] = self.nk_edges - 1

        l = l.view(-1, self.nd_nodes, self.bw)
        for i in range(num_samples):
            a[i] = unflatten(l[i])

        print("-----------------------------------")
        print(samp_n[0])
        print(samp_b[0])
        print(x[0])
        print(a[0])

        return x, a


class MolSPNBandSort(MolSPNBandCore):
    def __init__(self,
                 nc,
                 nd_n,
                 nk_n,
                 nk_e,
                 nl_n,
                 nl_e,
                 nr_n,
                 nr_e,
                 ns_n,
                 ns_e,
                 ni_n,
                 ni_e,
                 bw,
                 device='cuda'):
        super().__init__(nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, bw, device)

    def _forward(self, x, a):
        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(a.view(-1, self.nd_edges))
        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.logits_w, dim=1), dim=1)

MODELS = {
    'molspn_band_sort': MolSPNBandSort,
}

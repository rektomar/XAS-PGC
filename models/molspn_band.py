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

        self.logits_n = nn.Parameter(torch.ones(nd_n,  device=device), requires_grad=True)
        self.logits_b = nn.Parameter(torch.ones(bw+1,  device=device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.ones(1, nc, device=device), requires_grad=True)

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, xx, aa, num_full):
        pass

    def forward(self, x, a):
        l = []
        mask_nodes = (x != self.nk_nodes-1)
        mask_bands = (a == self.nk_edges-1).all(dim=1)
        n_nodes = torch.count_nonzero( mask_nodes, dim=1)
        n_bands = torch.count_nonzero(~mask_bands, dim=1)
        d_nodes = torch.distributions.Categorical(logits=self.logits_n)
        d_bands = torch.distributions.Categorical(logits=self.logits_b)

        for n in torch.unique(n_nodes):
            marginalize_nodes(self.network_nodes, self.nd_nodes, n)
            marginalize_edges(self.network_edges, self.nd_nodes, self.bw, n)
            l.append(self._forward(x[n_nodes == n], a[n_nodes == n], n))

        n, i = n_nodes.sort()
        m = n_bands[i]
        n = n.to(self.device) - 1
        m = m.to(self.device)

        return d_nodes.log_prob(n) + d_bands.log_prob(m) + torch.cat(l)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        o = 0
        x = torch.zeros(num_samples, self.nd_nodes)
        l = torch.zeros(num_samples, self.nd_nodes, self.bw)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes)

        d_n = torch.distributions.Categorical(logits=self.logits_n)
        d_b = torch.distributions.Categorical(logits=self.logits_b)
        d_w = torch.distributions.Categorical(logits=self.logits_w)

        n_nodes = d_n.sample((num_samples, )).to(torch.int)

        for n, n_count in zip(*torch.unique(n_nodes, return_counts=True)):
            marginalize_nodes(self.network_nodes, self.nd_nodes, n+1)
            marginalize_edges(self.network_edges, self.nd_nodes, self.bw, n+1)

            components = d_w.sample((n_count, ))
            bandwidths = d_b.sample((n_count, ))
            for i, (c, b) in enumerate(zip(components, bandwidths)):
                x[i+o] = self.network_nodes.sample(1, class_idx=c).cpu()
                l[i+o] = self.network_edges.sample(1, class_idx=c).view(-1, self.nd_nodes, self.bw).cpu()
                x[i+o, n:] = self.nk_nodes - 1
                # print("-------------------")
                # print(b)
                # print(l[i+o])
                l[i+o, :, b:] = self.nk_edges - 1
                # print(l[i+o])
            o += n_count

        for i in range(num_samples):
            a[i] = unflatten(l[i])

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

    def _forward(self, x, a, num_full):
        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(a.view(-1, self.nd_edges))
        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.logits_w, dim=1), dim=1)

MODELS = {
    'molspn_band_sort': MolSPNBandSort,
}

import math
import torch
import torch.nn as nn
import itertools

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from torch.distributions import Poisson
from models.utils import ohe2cat, cat2ohe
from tqdm import tqdm


def marginalize_nodes(network, nd_nodes, num_empty, num_full):
    with torch.no_grad():
        if num_empty > 0:
            mx = torch.zeros(nd_nodes, dtype=torch.bool)
            mx[num_full:] = True
            marginalization_idx = torch.arange(nd_nodes)[mx.view(-1)]

            network.set_marginalization_idx(marginalization_idx)
        else:
            network.set_marginalization_idx(None)

def marginalize_edges(network, nd_nodes, num_empty, num_full):
    with torch.no_grad():
        if num_empty > 0:
            ma = torch.zeros(nd_nodes, nd_nodes, dtype=torch.bool)
            ma[num_full:, :] = True
            ma[:, num_full:] = True
            marginalization_idx = torch.arange(nd_nodes * (nd_nodes - 1) // 2)[ma[torch.tril(torch.ones(nd_nodes, nd_nodes), diagonal=-1) == 1]]

            network.set_marginalization_idx(marginalization_idx)
        else:
            network.set_marginalization_idx(None)


class MolSPNMargCore(nn.Module):
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
                 device='cuda'):
        super().__init__()
        nd_e = nd_n * (nd_n - 1) // 2
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

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

        # self.rate = nn.Parameter(torch.randn(1, device=device), requires_grad=True)
        self.logits = nn.Parameter(torch.ones(nd_n, device=device), requires_grad=True)
        self.weights = nn.Parameter(torch.log_softmax(torch.randn(1, nc, device=device), dim=1), requires_grad=True)
        self.m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, xx, aa, num_full):
        pass

    def forward(self, x, a):
        l = []
        x, a = ohe2cat(x, a)
        c = torch.count_nonzero(x == self.nk_nodes-1, dim=1)
        for num_empty in torch.unique(c):
            num_full = self.nd_nodes-num_empty.item()
            marginalize_nodes(self.network_nodes, self.nd_nodes, num_empty.item(), num_full)
            marginalize_edges(self.network_edges, self.nd_nodes, num_empty.item(), num_full)
            l.append(self._forward(x[c == num_empty], a[c == num_empty], num_full))

        num_empty, _ = c.sort()
        n = self.nd_nodes - num_empty - 1
        # d = Poisson(self.rate.exp())
        d = torch.distributions.Categorical(logits=self.logits)

        return d.log_prob(n.to(self.device)) + torch.cat(l)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        o = 0
        x = torch.zeros(num_samples, self.nd_nodes)
        l = torch.zeros(num_samples, self.nd_edges)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes)

        # d = Poisson(self.rate.exp())
        d = torch.distributions.Categorical(logits=self.logits)
        n = self.nd_nodes - d.sample((num_samples, )).clamp(0, self.nd_nodes).to(torch.int)
        for num_empty, num_samples in zip(*torch.unique(n, return_counts=True)):
            num_full = self.nd_nodes-num_empty.item()
            marginalize_nodes(self.network_nodes, self.nd_nodes, num_empty, num_full)
            marginalize_edges(self.network_edges, self.nd_nodes, num_empty, num_full)

            cs = torch.distributions.Categorical(logits=self.weights).sample((num_samples, ))
            for i, c in enumerate(cs):
                x[i+o, :] = self.network_nodes.sample(1, class_idx=c).cpu()
                l[i+o, :] = self.network_edges.sample(1, class_idx=c).cpu()
            o += num_samples

        a[:, self.m] = l

        return cat2ohe(x, a, self.nk_nodes, self.nk_edges)


class MolSPNMargSort(MolSPNMargCore):
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
                 device='cuda'):
        super().__init__(nc, nd_n, nk_n, nk_e, nl_n, nl_e, nr_n, nr_e, ns_n, ns_e, ni_n, ni_e, device)

    def _forward(self, x, a, num_full):
        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(a[:, self.m].view(-1, self.nd_edges))
        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.weights, dim=1), dim=1)

MODELS = {
    'molspn_marg_sort': MolSPNMargSort,
}

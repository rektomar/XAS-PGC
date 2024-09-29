import math
import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from models.utils import ohe2cat, cat2ohe


def num_bits(n):
    return int((math.log(n-1)/math.log(2)) + 1)

def _cat2bin(x, n):
    mask = 2**torch.arange(num_bits(n)).type_as(x)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

def _bin2cat(b, n):
    mask = 2**torch.arange(num_bits(n)).type_as(b)
    return torch.sum(mask*b, -1)

def cat2bin(x, a, num_node_types, num_edge_types):
    x = _cat2bin(x, num_node_types)
    a = _cat2bin(a, num_edge_types)
    return x, a

def bin2cat(x, a, num_node_types, num_edge_types):
    x = _bin2cat(x, num_node_types)
    a = _bin2cat(a, num_edge_types)
    return x, a


class MolSPNNoneCore(nn.Module):
    def __init__(self,
                 nc,
                 nd_n,
                 nd_e,
                 nk_n,
                 nk_e,
                 ns_n,
                 ns_e,
                 ni_n,
                 ni_e,
                 graph_nodes,
                 graph_edges,
                 device
                 ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

        args_nodes = EinsumNetwork.Args(
            num_var=nd_n,
            num_dims=num_bits(nk_n),
            num_input_distributions=ni_n,
            num_sums=ns_n,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.BinomialArray,
            exponential_family_args={'N': 1},
            use_em=False)
        args_edges = EinsumNetwork.Args(
            num_var=nd_e,
            num_dims=num_bits(nk_e),
            num_input_distributions=ni_e,
            num_sums=ns_e,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.BinomialArray,
            exponential_family_args={'N': 1},
            use_em=False)

        self.network_nodes = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
        self.network_edges = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
        self.network_nodes.initialize()
        self.network_edges.initialize()

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, x, a):
        pass

    def forward(self, x, a):
        x, a = cat2bin(x, a, self.nk_nodes, self.nk_edges)
        return self._forward(x, a)

    def logpdf(self, x, a):
        return self(x, a).mean()

    @abstractmethod
    def _sample(self, num_samples):
        pass

    def sample(self, num_samples):
        x, a = self._sample(num_samples)
        x, a = bin2cat(x, a, self.nk_nodes, self.nk_edges)
        x = x.clamp(max=self.nk_nodes-1)
        a = a.clamp(max=self.nk_edges-1)
        return x, a


class MolSPNNoneSort(MolSPNNoneCore):
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
                 device='cuda'
                 ):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(nc, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device)

        self.weights = nn.Parameter(torch.log_softmax(torch.randn(1, nc, device=self.device), dim=1), requires_grad=True)
        self.m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)

    def _forward(self, x, a):
        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(a[:, self.m].view(-1, self.nd_edges, num_bits(self.nk_edges)))
        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.weights, dim=1), dim=1)

    def _sample(self, num_samples):
        x = torch.zeros(num_samples, self.nd_nodes, num_bits(self.nk_nodes))
        l = torch.zeros(num_samples, self.nd_edges, num_bits(self.nk_edges))

        cs = torch.distributions.Categorical(logits=self.weights).sample((num_samples, ))
        for i, c in enumerate(cs):
            x[i, :, :] = self.network_nodes.sample(1, class_idx=c).cpu()
            l[i, :, :] = self.network_edges.sample(1, class_idx=c).cpu()

        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, num_bits(self.nk_edges))
        a[:, self.m, :] = l

        return x, a

MODELS = {
    'molspn_none_sort': MolSPNNoneSort,
}

if __name__ == '__main__':
    bs = 3
    nx = 2
    nk_n = 5
    nk_e = 4

    x = torch.randint(0, nk_n, (bs, nx))
    a = torch.randint(0, nk_e, (bs, nx, nx))

    b = _cat2bin(x, nk_n)
    print(x)
    print(b[0, :, 0])
    print(b[0, :, 1])
    print(b[0, :, 2])
    print(b[1, :, 0])
    print(b[1, :, 1])
    print(b[1, :, 2])
    print(b[2, :, 0])
    print(b[2, :, 1])
    print(b[2, :, 2])
    b_rec = _bin2cat(b, nk_n)
    print(b_rec)

    print(x.size())
    print(b.size())
    print(b_rec.size())

    b = _cat2bin(a, nk_e)
    print(a)
    print(b[0, :, :, 0])
    print(b[0, :, :, 1])
    print(b[1, :, :, 0])
    print(b[1, :, :, 1])
    print(b[2, :, :, 0])
    print(b[2, :, :, 1])
    b_rec = _bin2cat(b, nk_e)
    print(b_rec)

    print(a.size())
    print(b.size())
    print(b_rec.size())

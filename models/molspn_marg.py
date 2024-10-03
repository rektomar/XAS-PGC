import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray


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

        self.logits_n = nn.Parameter(torch.randn( nd_n, device=device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.randn(1, nc, device=device), requires_grad=True)
        self.m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool, device=device), diagonal=-1)

        self.device = device
        self.to(device)

    @abstractmethod
    def _forward(self, xx, aa, num_full):
        pass

    def forward(self, x, a):
        m_nodes = (x != self.nk_nodes-1)
        m_edges = (m_nodes.unsqueeze(2) * m_nodes.unsqueeze(1))[:, self.m].view(-1, self.nd_edges)

        n = m_nodes.sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)

        # self.network_nodes.set_marginalization_mask(m_nodes)
        # self.network_edges.set_marginalization_mask(m_edges)

        return dist_n.log_prob(n) + self._forward(x, a)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        # if num_samples > 200:
        #     num_samples = 200
        # x = torch.zeros(num_samples, self.nd_nodes,                device=self.device)
        # l = torch.zeros(num_samples, self.nd_edges,                device=self.device)
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, device=self.device)

        # x += self.nk_nodes - 1
        # l += self.nk_edges - 1
        a += self.nk_edges - 1

        dist_n = torch.distributions.Categorical(logits=self.logits_n)
        dist_w = torch.distributions.Categorical(logits=self.logits_w)
        samp_n = dist_n.sample((num_samples, ))
        samp_w = dist_w.sample((num_samples, )).squeeze()

        m_nodes = torch.arange(self.nd_nodes, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        m_edges = (m_nodes.unsqueeze(2) * m_nodes.unsqueeze(1))[:, self.m].view(-1, self.nd_edges)

        x = self.network_nodes.sample(num_samples, class_idxs=samp_w)
        l = self.network_edges.sample(num_samples, class_idxs=samp_w)
        x[~m_nodes] = self.nk_nodes - 1
        l[~m_edges] = self.nk_edges - 1

        a[:, self.m] = l
        a.transpose(1,2)[:, self.m] = l

        return x.cpu(), a.cpu()


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

    def _forward(self, x, a):
        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(a[:, self.m].view(-1, self.nd_edges))
        return torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.logits_w, dim=1), dim=1)

MODELS = {
    'molspn_marg_sort': MolSPNMargSort,
}

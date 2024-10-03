import torch
import torch.nn as nn
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray


class MolSPNPoonSort(nn.Module):
    def __init__(self, nc, nd_n, nk_n, nk_e, nl_n, nr_n, ns_n, ns_e, ni_n, ni_e, num_pieces, device='cuda'):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_n**2
        self.nk_nodes = nk_n
        self.nk_edges = nk_e

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.poon_domingos_structure(shape=[nd_n, nd_n], delta=[[nd_n / d, nd_n / d] for d in num_pieces])

        args_nodes = EinsumNetwork.Args(
            num_var=nd_n,
            num_input_distributions=ni_n,
            num_sums=ns_n,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk_n},
            use_em=False)
        args_edges = EinsumNetwork.Args(
            num_var=nd_n**2,
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
        self.m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool), diagonal=-1)

        self.device = device
        self.to(device)

    def forward(self, x, a):
        m_nodes = (x != self.nk_nodes-1)

        n = m_nodes.sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)

        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(a.view(-1, self.nd_edges))

        return dist_n.log_prob(n) + torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.logits_w, dim=1), dim=1)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, device=self.device)

        dist_n = torch.distributions.Categorical(logits=self.logits_n)
        dist_w = torch.distributions.Categorical(logits=self.logits_w)
        samp_n = dist_n.sample((num_samples, ))
        samp_w = dist_w.sample((num_samples, )).squeeze()

        m_nodes = torch.arange(self.nd_nodes, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        m_edges = m_nodes.unsqueeze(2) * m_nodes.unsqueeze(1)

        x = self.network_nodes.sample(num_samples, class_idxs=samp_w)
        l = self.network_edges.sample(num_samples, class_idxs=samp_w).view(-1, self.nd_nodes, self.nd_nodes)
        x[~m_nodes] = self.nk_nodes - 1
        l[~m_edges] = self.nk_edges - 1

        a[:, self.m] = l[:, self.m]
        a.transpose(1,2)[:, self.m] = l[:, self.m]

        return x, a

MODELS = {
    'molspn_poon_sort': MolSPNPoonSort,
}
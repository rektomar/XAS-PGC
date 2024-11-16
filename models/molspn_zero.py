import torch
import torch.nn as nn

from einsum import Graph, EinsumNetwork, ExponentialFamilyArray


class MolSPNZeroSort(nn.Module):
    def __init__(self,
                 loader_trn,
                 nc,
                 nl_n,
                 nl_e,
                 nr_n,
                 nr_e,
                 ns_n,
                 ns_e,
                 ni_n,
                 ni_e,
                 device
                 ):
        super().__init__()
        x = torch.stack([b['x'] for b in loader_trn.dataset])
        a = torch.stack([b['a'] for b in loader_trn.dataset])

        self.nd_nodes = x.size(1)
        self.nd_edges = a.size(1)
        self.nk_nodes = len(x.unique())
        self.nk_edges = len(a.unique())

        args_nodes = EinsumNetwork.Args(
            num_var=self.nd_nodes,
            num_dims=1,
            num_input_distributions=ni_n,
            num_sums=ns_n,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': self.nk_nodes},
            use_em=False)
        args_edges = EinsumNetwork.Args(
            num_var=self.nd_edges,
            num_dims=1,
            num_input_distributions=ni_e,
            num_sums=ns_e,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': self.nk_edges},
            use_em=False)

        graph_nodes = Graph.random_binary_trees(self.nd_nodes, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(self.nd_edges, nl_e, nr_e)

        self.network_nodes = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
        self.network_edges = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
        self.network_nodes.initialize()
        self.network_edges.initialize()

        self.logits_n = nn.Parameter(torch.randn(self.nd_nodes, device=device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.randn(1,         nc, device=device), requires_grad=True)
        self.m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool, device=device), diagonal=-1)

        self.device = device
        self.to(device)

    def forward(self, x, a):
        n = (x > 0).sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)

        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(a[:, self.m].view(-1, self.nd_edges))

        return dist_n.log_prob(n) + torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.logits_w, dim=1), dim=1)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        dist_n = torch.distributions.Categorical(logits=self.logits_n)
        dist_w = torch.distributions.Categorical(logits=self.logits_w)
        samp_n = dist_n.sample((num_samples, ))
        samp_w = dist_w.sample((num_samples, )).squeeze()

        m_nodes = torch.arange(self.nd_nodes, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        m_edges = (m_nodes.unsqueeze(2) * m_nodes.unsqueeze(1))[:, self.m].view(-1, self.nd_edges)

        x = self.network_nodes.sample(num_samples, class_idxs=samp_w)
        l = self.network_edges.sample(num_samples, class_idxs=samp_w)
        x[~m_nodes] = 0
        l[~m_edges] = 0

        a = torch.zeros((num_samples, self.nd_nodes, self.nd_nodes), device=self.device)
        a[:, self.m] = l
        a.transpose(1, 2)[:, self.m] = l

        return x.cpu(), a.cpu()

MODELS = {
    'molspn_zero_sort': MolSPNZeroSort,
}

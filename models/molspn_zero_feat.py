import os
import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from models.utils import ohe2cat, cat2ohe


class MolSPNZeroFeatCore(nn.Module):
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
                 device,
                 regime,
                 dc_n=0,
                 dc_e=0
                 ):
        super().__init__()
        self.nd_nodes = nd_n
        self.nd_edges = nd_e
        self.nk_nodes = nk_n
        self.nk_edges = nk_e
        self.regime = regime

        if regime == 'cat':
            ef_dist_n = ExponentialFamilyArray.CategoricalArray
            ef_dist_e = ExponentialFamilyArray.CategoricalArray
            ef_args_n = {'K': nk_n}
            ef_args_e = {'K': nk_e}
            num_dim_n = 1
            num_dim_e = 1
        elif regime == 'bin':
            ef_dist_n = ExponentialFamilyArray.BinomialArray
            ef_dist_e = ExponentialFamilyArray.BinomialArray
            ef_args_n = {'N': nk_n-1}
            ef_args_e = {'N': nk_e-1}
            num_dim_n = 1
            num_dim_e = 1
        elif regime == 'deq':
            ef_dist_n = ExponentialFamilyArray.NormalArray
            ef_dist_e = ExponentialFamilyArray.NormalArray
            ef_args_n = {'min_var': 1e-1, 'max_var': 1e-0}
            ef_args_e = {'min_var': 1e-1, 'max_var': 1e-0}
            num_dim_n = nk_n
            num_dim_e = nk_e
            self.dc_n = dc_n
            self.dc_e = dc_e
        else:
            os.error('Unsupported \'regime\'.')

        args_nodes = EinsumNetwork.Args(
            num_var=nd_n,
            num_dims=num_dim_n,
            num_input_distributions=ni_n,
            num_sums=ns_n,
            num_classes=nc,
            exponential_family=ef_dist_n,
            exponential_family_args=ef_args_n,
            use_em=False)
        args_edges = EinsumNetwork.Args(
            num_var=nd_e,
            num_dims=num_dim_e,
            num_input_distributions=ni_e,
            num_sums=ns_e,
            num_classes=nc,
            exponential_family=ef_dist_e,
            exponential_family_args=ef_args_e,
            use_em=False)

        self.network_nodes = EinsumNetwork.EinsumNetwork(graph_nodes, args_nodes)
        self.network_edges = EinsumNetwork.EinsumNetwork(graph_edges, args_edges)
        self.network_nodes.initialize()
        self.network_edges.initialize()

        self.to(device)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @abstractmethod
    def _forward(self, x, a, m_nodes):
        pass

    def prepare_input(self, x: torch.Tensor, a: torch.Tensor):
        if   self.regime == 'cat' or self.regime == 'bin':
            _x, _a = x, a
        elif self.regime == 'deq':
            x, a = cat2ohe(x, a, self.nk_nodes, self.nk_edges)
            _x = x + self.dc_n*torch.rand(x.size(), device=self.device)
            _a = a + self.dc_e*torch.rand(a.size(), device=self.device)
        else:
            os.error('Unknown regime')
        
        return _x, _a

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        m_nodes = (x != self.nk_nodes-1)
        _x, _a = self.prepare_input(x, a)
        ll_card, ll_nodes, ll_edges, logits_w = self._forward(_x, _a, m_nodes)
        return ll_card + torch.logsumexp(ll_nodes + ll_edges + logits_w, dim=1)

    def logpdf(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self(x, a).mean()

    @abstractmethod
    def sample(self, num_samples: int):
        pass


class MolSPNZeroSortFeat(MolSPNZeroFeatCore):
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
                 regime,
                 dc_n=0.6,
                 dc_e=0.6,
                 device='cuda'
                 ):
        nd_e = nd_n * (nd_n - 1) // 2

        graph_nodes = Graph.random_binary_trees(nd_n, nl_n, nr_n)
        graph_edges = Graph.random_binary_trees(nd_e, nl_e, nr_e)

        super().__init__(nc, nd_n, nd_e, nk_n, nk_e, ns_n, ns_e, ni_n, ni_e, graph_nodes, graph_edges, device, regime, dc_n, dc_e)

        self.logits_n = nn.Parameter(torch.randn( nd_n, device=device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.randn(1, nc, device=device), requires_grad=True)
        self.m = torch.tril(torch.ones(self.nd_nodes, self.nd_nodes, dtype=torch.bool, device=device), diagonal=-1)

    def _prepare_edges(self, a: torch.Tensor) -> torch.Tensor:
        if   self.regime == 'cat' or self.regime == 'bin':
            _a = a[:, self.m].view(-1, self.nd_edges)
        elif self.regime == 'deq':
            _a = a[:, self.m, :].view(-1, self.nd_edges, self.nk_edges)
        else:
            os.error('Unknown regime')
        return _a

    def _forward(self, x: torch.Tensor, a: torch.Tensor, m_nodes: torch.Tensor):
        n = m_nodes.sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)
        ll_card = dist_n.log_prob(n)

        _a = self._prepare_edges(a)
        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(_a)

        return ll_card, ll_nodes, ll_edges, torch.log_softmax(self.logits_w, dim=1)

    def sample(self, num_samples: int):
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

        if   self.regime == 'cat' or self.regime == 'bin':
            a = torch.full((num_samples, self.nd_nodes, self.nd_nodes), self.nk_edges - 1, dtype=torch.float, device=self.device)
            a[:, self.m] = l
            a.transpose(1, 2)[:, self.m] = l
        elif self.regime == 'deq':
            a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, self.nk_edges, device=self.device)
            a[:, self.m] = l
            x, a = ohe2cat(x, a)
        else:
            os.error('Unknown regime')

        return x.cpu(), a.cpu()

MODELS = {
    'molspn_zero_sort_feat': MolSPNZeroSortFeat,
}

if __name__ == "__main__":
    import json
    with open(f'config/qm9/molspn_zero_sort_feat.json', 'r') as f:
        hyperpars = json.load(f)

    model = MolSPNZeroSortFeat(**hyperpars['model_hyperpars'])
    x, a = model.sample(32)
    x, a = x.to(model.device), a.to(model.device)

    print(model(x, a))

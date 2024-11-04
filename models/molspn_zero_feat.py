import os
import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from models.utils import ohe2cat, cat2ohe

from torch.distributions import Normal, Categorical, MixtureSameFamily
from torch.nn.functional import softplus


class Normal_(nn.Module):

    def __init__(self, nc: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.m = nn.Parameter(torch.randn(nc))
        self.s = nn.Parameter(torch.randn(nc))

    @property
    def distribution(self):
        return Normal(self.m, softplus(self.s)+self.eps)
    
    def logpdf(self, y):
        return self.distribution.log_prob(y)
    
    def forward(self, y):
        return self.logpdf(y)
    
    def sample(self, n_samples: int):
        return self.distribution.sample((n_samples,))
    

class GMM_(nn.Module):

    def __init__(self, nc: int, num_comps: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.m = nn.Parameter(torch.randn(nc, num_comps))
        self.s = nn.Parameter(torch.randn(nc, num_comps))
        self.w = nn.Parameter(torch.randn(num_comps))

    @property
    def distribution(self):
        mix = Categorical(logits=self.w)
        comp = Normal(self.m, softplus(self.s)+self.eps)
        return MixtureSameFamily(mix, comp)
    
    def logpdf(self, y):
        return self.distribution.log_prob(y)
    
    def forward(self, y):
        return self.logpdf(y)
    
    def sample(self, n_samples: int):
        return self.distribution.sample((n_samples,))


class ExtraFeatures(nn.Module):

    def __init__(self, nc: int):
        super().__init__()

        # nd = 1
        # num_comps = 4
        # self.distribution = GMMF(nc, num_comps)

        self.distribution = Normal_(nc)

    def forward(self, y):
        return self.distribution.logpdf(y)
        
    def sample(self, n_samples):
        return self.distribution.sample(n_samples)


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

        self.network_features = ExtraFeatures(nc)

        self.to(device)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @abstractmethod
    def _forward(self, x, a, m_nodes):
        pass

    def _reshape_edges(self, a: torch.Tensor) -> torch.Tensor:
        if   self.regime == 'cat' or self.regime == 'bin':
            _a = a[:, self.m].view(-1, self.nd_edges)
        elif self.regime == 'deq':
            _a = a[:, self.m, :].view(-1, self.nd_edges, self.nk_edges)
        else:
            os.error('Unknown regime')
        return _a
    
    def _reshape_edge_mask(self, ma: torch.Tensor) -> torch.Tensor:
        if   self.regime == 'cat' or self.regime == 'bin':
            _ma = ma[:, self.m].view(-1, self.nd_edges)
        elif self.regime == 'deq':
            _ma = ma[:, self.m, :].view(-1, self.nd_edges, 1).expand(-1, -1,self.nk_edges)
        else:
            os.error('Unknown regime')
        return _ma

    def prepare_input(self, x: torch.Tensor, a: torch.Tensor):
        if   self.regime == 'cat' or self.regime == 'bin':
            _x, _a = x, a
        elif self.regime == 'deq':
            x, a = cat2ohe(x, a, self.nk_nodes, self.nk_edges)
            _x = x + self.dc_n*torch.rand(x.size(), device=self.device)
            _a = a + self.dc_e*torch.rand(a.size(), device=self.device)
        else:
            os.error('Unknown regime')
        
        _a = self._reshape_edges(_a) 
        return _x, _a

    def forward(self, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.network_nodes.set_marginalization_mask(None)
        self.network_edges.set_marginalization_mask(None)

        m_nodes = (x != self.nk_nodes-1)
        _x, _a = self.prepare_input(x, a)
        ll_card, ll_nodes, ll_edges, logits_w = self._forward(_x, _a, m_nodes)
        ll_feat = self.network_features(y)
        return ll_card + torch.logsumexp(ll_nodes + ll_edges + ll_feat + logits_w, dim=1)

    def logpdf(self, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self(x, a, y).mean()

    @abstractmethod
    def sample(self, num_samples: int):
        pass

    @abstractmethod
    def sample_conditional(self, x: torch.Tensor, a: torch.Tensor, m_x: torch.Tensor, m_a: torch.Tensor, num_samples: int):
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

    def _forward(self, x: torch.Tensor, a: torch.Tensor, m_nodes: torch.Tensor):
        n = m_nodes.sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)
        ll_card = dist_n.log_prob(n)

        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(a)

        return ll_card, ll_nodes, ll_edges, torch.log_softmax(self.logits_w, dim=1)
    
    def logpdf_features(self, y):
        ll_features = self.network_features(y)
        return torch.logsumexp(ll_features + torch.log_softmax(self.logits_w, dim=1), dim=1)
    
    def predict_features(self, x, a):
        # returns E[y|G], where y are additional features/properties 
        # and G is graph induced by (x, a)
        x = x.to(self.device)
        a = a.to(self.device)

        self.network_nodes.set_marginalization_mask(None)
        self.network_edges.set_marginalization_mask(None)
        m_nodes = (x != self.nk_nodes-1)
        _x, _a = self.prepare_input(x, a)
        _, ll_nodes, ll_edges, logits_w = self._forward(_x, _a, m_nodes)

        logits_w = ll_nodes + ll_edges + logits_w
        prior = Categorical(logits=logits_w)
        comps = self.network_features.distribution.distribution

        distr = MixtureSameFamily(prior, comps)
        return distr.mean
    
    def predict_features_marginal(self, x, a, mx, ma):
        # returns E[y|G_o], where y are additional features/properties 
        # and G_o is subgraph of G = (x, a) induced by corresponing masks (mx, ma)
        x, a = x.to(self.device), a.to(self.device)
        mx, ma = mx.to(self.device), ma.to(self.device)

        _mx, _ma = mx, self._reshape_edge_mask(ma)
        self.network_nodes.set_marginalization_mask(_mx)
        self.network_edges.set_marginalization_mask(_ma)
        m_nodes = (x != self.nk_nodes-1)
        _x, _a = self.prepare_input(x, a)
        _, ll_nodes, ll_edges, logits_w = self._forward(_x, _a, m_nodes)

        logits_w = ll_nodes + ll_edges + logits_w
        prior = Categorical(logits=logits_w)
        comps = self.network_features.distribution.distribution

        distr = MixtureSameFamily(prior, comps)
        return distr.mean
    
    def _sample(self, logits_w: torch.Tensor, logits_n: torch.Tensor, num_samples: int, xo=None, ao=None):
        dist_n = torch.distributions.Categorical(logits=logits_n)
        dist_w = torch.distributions.Categorical(logits=logits_w)
        samp_n = dist_n.sample((num_samples, ))
        samp_w = dist_w.sample((num_samples, )).squeeze(-1)

        m_nodes = torch.arange(self.nd_nodes, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        m_edges = (m_nodes.unsqueeze(2) * m_nodes.unsqueeze(1))[:, self.m].view(-1, self.nd_edges)

        # print('masks', m_nodes.shape, m_edges.shape)
        x = self.network_nodes.sample(num_samples, class_idxs=samp_w, x=xo)
        l = self.network_edges.sample(num_samples, class_idxs=samp_w, x=ao)
        # print('output_samples', num_samples, x.shape, l.shape)
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

        return x, a

    def sample(self, num_samples: int):
        self.network_nodes.set_marginalization_mask(None)
        self.network_edges.set_marginalization_mask(None)
        x, a = self._sample(self.logits_w, self.logits_n, num_samples)
        return x.cpu(), a.cpu()
     
    @torch.no_grad
    def sample_conditional(self, x: torch.Tensor, a: torch.Tensor, mx: torch.Tensor, ma: torch.Tensor, num_samples: int):
        # for single observation only
        assert (len(x)==1) and (len(a)==1), "Cond. sampling is implemented only for single input observation."
        assert mx.sum() < self.nd_nodes, "Chosen subgraph is too big."
        _mx, _ma = mx, self._reshape_edge_mask(ma)

        # calculate adjusted mixture weights
        m_nodes = (x != self.nk_nodes-1)
        self.network_nodes.set_marginalization_mask(_mx)
        self.network_edges.set_marginalization_mask(_ma)
        _x, _a = self.prepare_input(x, a)
        _, ll_nodes, ll_edges, logits_w = self._forward(_x, _a, m_nodes)
        logits_w = logits_w + ll_nodes + ll_edges

        logits_n = self.logits_n.masked_fill(_mx.squeeze(), -torch.inf)

        xo = _x.float().expand(num_samples, -1)
        ao = _a.float().expand(num_samples, -1)
        self.network_nodes.set_marginalization_mask(_mx.expand(num_samples, -1))
        self.network_edges.set_marginalization_mask(_ma.expand(num_samples, -1))
        xc, ac = self._sample(logits_w, logits_n, num_samples, xo=xo, ao=ao)
        return xc.cpu(), ac.cpu()

    def sample_given_features(self, y: torch.Tensor, num_samples: int):
        # G ~ p(G | y), G = (X, A)   
        self.network_nodes.set_marginalization_mask(None)
        self.network_edges.set_marginalization_mask(None)
        logits_w = torch.log_softmax(self.logits_w, -1) + self.network_features(y)
        x, a = self._sample(logits_w, self.logits_n, num_samples)
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

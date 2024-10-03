import torch
import torch.nn as nn

from abc import abstractmethod
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from models.utils import ohe2cat, cat2ohe, EncoderFFNN
from models.graph_transformer import GraphTransformer


def permute_x(x, p):
    return torch.matmul(p, x)

def permute_a(a, p):
    a = torch.matmul(p.unsqueeze(1), a)
    a = torch.matmul(p.unsqueeze(1), a.permute(0, 2, 1, 3))
    return a.permute(0, 2, 1, 3)

def permute_g(x, a, p):
    x = permute_x(x, p)
    a = permute_a(a, p)
    return x, a

# Permuter and Penalty classes are based on https://github.com/jrwnter/pigvae

class Penalty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def entropy(p, axis, normalize=True, eps=10e-12):
        if normalize:
            p = p / (p.sum(axis=axis, keepdim=True) + eps)
        e = - torch.sum(p * torch.clamp_min(torch.log(p), -100), axis=axis)
        return e

    def forward(self, p, eps=10e-12):
        p = p + eps
        entropy_cols = self.entropy(p, axis=1, normalize=False)
        entropy_rows = self.entropy(p, axis=2, normalize=False)
        return entropy_cols.mean() + entropy_rows.mean()

class Permuter(nn.Module):
    def __init__(self, nd):
        super().__init__()
        self.scoring_function = nn.Linear(nd, 1)

    def score(self, x, m):
        s = self.scoring_function(x)
        s = s.masked_fill(m.unsqueeze(-1) == 0, s.min().item() - 1)
        return s

    def soft_sort(self, s, hard, tau):
        s_sorted = s.sort(descending=True, dim=1)[0]
        pairwise_differences = (s.transpose(1, 2) - s_sorted).abs().neg() / tau
        p = pairwise_differences.softmax(-1)
        if hard:
            p_ = torch.zeros_like(p, device=p.device)
            p_.scatter_(-1, p.topk(1, -1)[1], value=1)
            p = (p_ - p).detach() + p
        return p_

    def mask_perm(self, p, m):
        bs, nd = m.shape
        e = torch.eye(nd, nd).unsqueeze(0).expand(bs, -1, -1).type_as(p)
        m = m.unsqueeze(-1).expand(-1, -1, nd)
        p = torch.where(m, p, e)
        return p

    def forward(self, h, m, hard=False, tau=0.001):
        h = h + torch.randn_like(h) * 0.05
        s = self.score(h, m)
        p = self.soft_sort(s, hard, tau)
        p = p.transpose(2, 1)
        p = self.mask_perm(p, m)
        return p



class MolSPNPermSort(nn.Module):
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

        # n_layers = 5
        # n_head = 8
        # mh_n = 128
        # mh_e = 64
        # mh_y = 0
        # nh_n = 128
        # nh_e = 64
        # nh_y = 0
        # df_n = 256
        # df_e = 128
        # df_y = 0
        # self.transformer = GraphTransformer(n_layers, nk_n, nk_e, 0, nk_n, nk_e, 0, mh_n, mh_e, mh_y, n_head, nh_n, nh_e, nh_y, df_n, df_e, df_y)

        h_x = 2048
        h_a = 1024
        h_y = 32
        l_x = 2
        l_a = 2
        l_y = 2
        l_b = 2

        self.encoder = EncoderFFNN(
            nd_n, nd_n, nk_n, nk_e, 0, nk_n, nk_e, 0, h_x, h_a, h_y, l_x, l_a, l_y, l_b, device=device
        )

        self.permuter = Permuter(nk_n)
        self.penalty = Penalty()

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

        x, a = cat2ohe(x, a, self.nk_nodes, self.nk_edges)
        # h = self.transformer(x, a, torch.zeros(x.size(0), 0).type_as(x), m_nodes)[0]
        h = self.encoder(x, a, torch.zeros(x.size(0), 0).type_as(x))[0]
        p = self.permuter(h, ~m_nodes)
        p = torch.nn.functional.one_hot(torch.argmax(p, -1), num_classes=x.shape[1]).to(torch.float)
        x, a = permute_g(x, a, p)
        x, a = ohe2cat(x, a)

        # self.network_nodes.set_marginalization_mask(m_nodes)
        # self.network_edges.set_marginalization_mask(m_edges)

        ll_nodes = self.network_nodes(x)
        ll_edges = self.network_edges(a[:, self.m].view(-1, self.nd_edges))

        return dist_n.log_prob(n) + self.penalty(p) + torch.logsumexp(ll_nodes + ll_edges + torch.log_softmax(self.logits_w, dim=1), dim=1)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        a = torch.zeros(num_samples, self.nd_nodes, self.nd_nodes, device=self.device)
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


MODELS = {
    'molspn_perm_sort': MolSPNPermSort,
}

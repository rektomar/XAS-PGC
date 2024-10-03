import torch
import torch.nn as nn

from torch.distributions import Categorical
from typing import Optional


def cat2ohe(x, a, num_node_types, num_edge_types):
    x = torch.nn.functional.one_hot(x.to(torch.int64), num_classes=num_node_types)
    a = torch.nn.functional.one_hot(a.to(torch.int64), num_classes=num_edge_types)
    return x.to(torch.float), a.to(torch.float)

def ohe2cat(x, a):
    x = torch.argmax(x, dim=-1)
    a = torch.argmax(a, dim=-1)
    return x.to(torch.float), a.to(torch.float)

    def __init__(self,
                 network: nn.Module,
                 num_chunks: Optional[int]=1,
                 device: Optional[str]='cuda'
                 ):
        super(CategoricalDecoder, self).__init__()
        self.network = network
        self.num_chunks = num_chunks
        self.device = device

    def forward(self,
                xx: torch.Tensor,                                                  # (bs, no, n_xo)
                xa: torch.Tensor,                                                  # (bs, no, no, n_ao)
                xy: torch.Tensor,                                                  # (bs, n_yo)
                zx: torch.Tensor,                                                  # (bs, ni, n_xi)
                za: torch.Tensor,                                                  # (bs, ni, ni, n_ai)
                zy: torch.Tensor                                                   # (bs, n_yi)
                ):
        xx, xa = ohe2cat(xx, xa)                                                   # (bs, ni), (bs, ni, ni)
        xx = xx.float()
        xa = xa.float()

        logit_x, logit_a, logit_y = self.network(zx, za, zy)
        log_prob_x = Categorical(logits=logit_x).log_prob(xx)
        log_prob_a = Categorical(logits=logit_a).log_prob(xa)
        log_prob = log_prob_x.sum(dim=1) + log_prob_a.sum(dim=(1, 2))

        return log_prob

    def cm_logpdf_marginal(self,
                x_node: torch.Tensor,                                            # (bs, nd_no, nk_no)
                x_edge: torch.Tensor,                                            # (bs, nd_no, nd_no, nk_eo)
                m_node: torch.Tensor,                                            # (bs, nd_no, nk_no)
                m_edge: torch.Tensor,                                            # (bs, nd_no, nd_no, nk_eo)
                z_node: torch.Tensor,                                            # (bs, nd_ni, nk_ni)
                z_edge: torch.Tensor,                                            # (bs, nd_ni, nd_ni, nk_ei)
                z_feat: torch.Tensor
                ):
        x_node, x_edge = ohe2cat(x_node, x_edge)                                 # (bs, nd_no), (bs, nd_no, nd_no)
        x_node = x_node.unsqueeze(1).float()                                     # (bs, 1, nd_no)
        x_edge = x_edge.unsqueeze(1).float()                                     # (bs, 1, nd_no, nd_no)
        m_node = m_node.unsqueeze(1)                                             # (bs, 1, nd_no)
        m_edge = m_edge.unsqueeze(1)                                             # (bs, 1, nd_no, nd_no)

        log_prob = torch.zeros(len(x_node), len(z_node), device=self.device)     # (bs, num_chunks*chunk_size)
        for c in torch.arange(len(z_node)).chunk(self.num_chunks):
            logit_node, logit_edge, _ = self.network(z_node[c, :], z_edge[c, :], z_feat[c, :])    # (chunk_size, nd_no, nk_no), (chunk_size, nd_no, nd_no, nk_eo)
            log_prob_node = Categorical(logits=logit_node).log_prob(x_node)*m_node      # (bs, chunk_size, nd_no)
            log_prob_edge = Categorical(logits=logit_edge).log_prob(x_edge)*m_edge      # (bs, chunk_size, nd_no, nd_no)
            log_prob[:, c] = log_prob_node.sum(dim=-1) + log_prob_edge.sum(dim=(-2, -1))

        return log_prob

    def sample(self,
               zx: torch.Tensor,                                                   # (bs, ni, n_xi)
               za: torch.Tensor,                                                   # (bs, ni, ni, n_ai)
               zy: torch.Tensor                                                    # (bs, n_yi)
               ):
        logit_x, logit_a, logit_y = self.network(zx, za, zy)                       # (bs, no, n_xo), (bs, no, no, n_ao), (bs, n_yo)
        xx = Categorical(logits=logit_x).sample()                                  # (bs, no)
        xa = Categorical(logits=logit_a).sample()                                  # (bs, no, no)
        xx, xa = cat2ohe(xx, xa, self.network.n_xo, self.network.n_ao)             # (bs, no, n_xo), (bs, no, no, n_ao)
        return xx.cpu(), xa.cpu()

    def sample_conditional(self,
               x:  torch.Tensor,                                                    # (bs, nd_no, nk_no)
               a:  torch.Tensor,                                                    # (bs, nd_no, nd_no, nk_eo)
               mx: torch.Tensor,                                                   # (bs, nd_no, nk_no)
               ma: torch.Tensor,                                                   # (bs, nd_no, nd_no, nk_eo)
               zx: torch.Tensor,                                                   # (bs, ni, n_xi)
               za: torch.Tensor,                                                   # (bs, ni, ni, n_ai)
               zy: torch.Tensor,
               logw:  torch.Tensor
               ):
        # interpreting decoder as continuous mixture
        # marginal probs for each component
        logpdf_marg = self.cm_logpdf_marginal(x, a, mx, ma, zx, za, zy) 
        k = Categorical(logits=logw+logpdf_marg).sample() # add num_samples here 
        xc, ac = self.sample(zx[k], za[k], zy[k])

        xc[mx] = x[mx].cpu()
        ac[ma] = a[ma].cpu()
        return xc, ac
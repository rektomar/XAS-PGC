import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
from typing import Optional
from models.utils import zero_diagonal


class NodeEdgeAttBlock(nn.Module):
    def __init__(self,
                 nh_n: int,
                 nh_e: int,
                 n_head: int
                 ):
        super().__init__()
        assert nh_n % n_head == 0, f"nh_n: {nh_n} -- nhead: {n_head}"
        self.nh_n = nh_n
        self.nh_e = nh_e
        self.nh_f = int(nh_n / n_head)
        self.n_head = n_head

        self.q = nn.Linear(nh_n, nh_n)
        self.k = nn.Linear(nh_n, nh_n)
        self.v = nn.Linear(nh_n, nh_n)

        self.a_add = nn.Linear(nh_e, nh_n)
        self.a_mul = nn.Linear(nh_e, nh_n)

        self.x_out = nn.Linear(nh_n, nh_n)
        self.a_out = nn.Linear(nh_n, nh_e)

    def forward(self, x, a):
        """
        :param x: (bs, n, nk_n)    node features
        :param a: (bs, n, n, nk_e) edge features
        :return: xhat, ahat of the same dims.
        """
        q = self.q(x)                          # (bs, n, nh_n)
        k = self.k(x)                          # (bs, n, nh_n)

        # Reshape to (bs, n, n_head, nh_f) with nh_n = n_head * nh_f
        q = q.reshape((q.size(0), q.size(1), self.n_head, self.nh_f))
        k = k.reshape((k.size(0), k.size(1), self.n_head, self.nh_f))

        q = q.unsqueeze(2)                     # (bs, 1, n, n_head, nh_f)
        k = k.unsqueeze(1)                     # (bs, n, 1, n head, nh_f)

        # Compute unnormalized attentions
        y = q * k                              # (bs, n, n, n_head, nh_f)
        y = y / math.sqrt(y.size(-1))

        # FiLM: Incorporate edge features to the self attention scores
        a1 = self.a_mul(a)                     # (bs, n, n, nh_n)
        a2 = self.a_add(a)                     # (bs, n, n, nh_n)
        a1 = a1.reshape((a.size(0), a.size(1), a.size(2), self.n_head, self.nh_f))
        a2 = a2.reshape((a.size(0), a.size(1), a.size(2), self.n_head, self.nh_f))
        y = y * (a1 + 1) + a2                  # (bs, n, n, n_head, nh_f)

        ahat = y.flatten(start_dim=3)          # (bs, n, n, nh_n)
        ahat = self.a_out(ahat)                # (bs, n, n, nh_n)

        # Compute attentions. 
        attn = torch.softmax(y, dim=2)         # (bs, n, n, n_head) ? (bs, n, n, n_head, nh_f)

        v = self.v(x)                          # (bs, n, nh_n
        v = v.reshape((v.size(0), v.size(1), self.n_head, self.nh_f))
        v = v.unsqueeze(1)                     # (bs, 1, n, n_head, nh_f)

        weighted_v = attn * v
        weighted_v = weighted_v.sum(dim=2)

        xhat = weighted_v.flatten(start_dim=2) # (bs, n, nh_n)
        xhat = self.x_out(xhat)

        return xhat, ahat

class TransformerLayer(nn.Module):
    """ Transformer that updates node and edge features
        nh_n: node features
        nh_e: edge features
        n_head: the number of heads in the multi_head_attention
        df_n: the dimension of the feedforward network for nodes
        df_e: the dimension of the feedforward network for edges
        dropout: dropout probablility
        eps: eps value in layer normalizations.
    """
    def __init__(self,
                 n_head: int,
                 nh_n: int,
                 nh_e: int,
                 df_n: int = 2048,
                 df_e: int = 128,
                 dropout: float = 0.1,
                 eps: float = 1e-5
                 ):
        super().__init__()
        self.self_attn = NodeEdgeAttBlock(nh_n, nh_e, n_head)

        self.lin_x1 = Linear(nh_n, df_n)
        self.lin_x2 = Linear(df_n, nh_n)
        self.norm_x1 = LayerNorm(nh_n, eps=eps)
        self.norm_x2 = LayerNorm(nh_n, eps=eps)
        self.dropout_x1 = Dropout(dropout)
        self.dropout_x2 = Dropout(dropout)
        self.dropout_x3 = Dropout(dropout)

        self.lin_a1 = Linear(nh_e, df_e)
        self.lin_a2 = Linear(df_e, nh_e)
        self.norm_a1 = LayerNorm(nh_e, eps=eps)
        self.norm_a2 = LayerNorm(nh_e, eps=eps)
        self.dropout_a1 = Dropout(dropout)
        self.dropout_a2 = Dropout(dropout)
        self.dropout_a3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, x: Tensor, a: Tensor):
        """ Pass the input through the encoder layer.
            x: (bs, n, d)
            a: (bs, n, n, d)
            Output: x, a of the same dims.
        """
        xhat, ahat = self.self_attn(x, a)

        x = self.norm_x1(x + self.dropout_x1(xhat))
        a = self.norm_a1(a + self.dropout_a1(ahat))

        ff_output_x = self.lin_x2(self.dropout_x2(self.activation(self.lin_x1(x))))
        ff_output_x = self.dropout_x3(ff_output_x)

        ff_output_a = self.lin_a2(self.dropout_a2(self.activation(self.lin_a1(a))))
        ff_output_a = self.dropout_a3(ff_output_a)

        x = self.norm_x2(x + ff_output_x)
        a = self.norm_a2(a + ff_output_a)

        return x, a

class GraphTransformer(nn.Module):
    def __init__(self,
                 n_layers: int,
                 nk_ni: int,
                 nk_ei: int,
                 nk_no: int,
                 nk_eo: int,
                 mh_n: int,
                 mh_e: int,
                 n_head: int,
                 nh_n: int,
                 nh_e: int,
                 df_n: int,
                 df_e: int,
                 act_fn_in = nn.ReLU,
                 act_fn_out = nn.ReLU,
                 device: Optional[str]='cuda'
                 ):
        super().__init__()
        self.n_layers = n_layers
        self.nk_ni = nk_ni
        self.nk_ei = nk_ei
        self.nk_no = nk_no
        self.nk_eo = nk_eo

        self.mlp_in_x = nn.Sequential(
            nn.Linear(nk_ni, mh_n),
            act_fn_in(),
            nn.Linear(mh_n, nh_n),
            act_fn_in()
        )
        self.mlp_in_a = nn.Sequential(
            nn.Linear(nk_ei, mh_e),
            act_fn_in(),
            nn.Linear(mh_e, nh_e),
            act_fn_in()
        )

        self.tf_layers = nn.ModuleList([
            TransformerLayer(
                n_head=n_head,
                nh_n=nh_n,
                nh_e=nh_e,
                df_n=df_n,
                df_e=df_e
            ) for _ in range(n_layers)
        ])

        self.mlp_out_x = nn.Sequential(
            nn.Linear(nh_n, mh_n),
            act_fn_out(),
            nn.Linear(mh_n, nk_no)
        )
        self.mlp_out_a = nn.Sequential(
            nn.Linear(nh_e, mh_e),
            act_fn_out(),
            nn.Linear(mh_e, nk_eo)
        )

        self.device = device

    def forward(self, x, a):
        # bs, n = x.shape[0], x.shape[1]

        # diag_mask = torch.eye(n)
        # diag_mask = ~diag_mask.type_as(a).bool()
        # diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        # x_to_out = x
        # a_to_out = a

        x = self.mlp_in_x(x)
        a = self.mlp_in_a(a)
        # a = (a + a.transpose(1, 2)) / 2

        for layer in self.tf_layers:
            x, a = layer(x, a)
            # a = (a + a.transpose(1, 2)) / 2

        x = self.mlp_out_x(x)
        a = self.mlp_out_a(a)

        # rep = x.size(-1)//x_to_out.size(-1)

        # x = (x + x_to_out.repeat(1, 1, rep))
        # a = (a + a_to_out.repeat(1, 1, 1, rep)) * diag_mask
        zero_diagonal(a, self.device)
        a = (a + a.transpose(1, 2)) / 2

        return x, a


if __name__ == '__main__':
    n_layers = 2
    nk_ni = 5
    nk_ei = 4
    mh_n = 256
    mh_e = 128
    n_head = 8
    nh_n = 256
    nh_e = 64
    df_n = 256
    df_e = 128

    model = GraphTransformer(n_layers, nk_ni, nk_ei, mh_n, mh_e, n_head, nh_n, nh_e, df_n, df_e)

    xx = torch.randn(100, 9, 5)
    aa = torch.randn(100, 9, 9, 4)

    x, a = model(xx, aa)
    print(x)
    print(a)

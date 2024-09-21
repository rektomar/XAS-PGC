import math

# The following code is based on https://github.com/cvignac/DiGress

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
from typing import Optional
from models.utils import set_diagonal

class Xtoy(nn.Module):
    def __init__(self, nh_n, nh_y):
        """ Map node features to global features
        """
        super().__init__()
        self.lin = nn.Linear(4 * nh_n, nh_y)

    def forward(self, x):
        """
        x: (bs, n, dx).
        """
        mm = x.mean(dim=1)
        mi = x.min(dim=1)[0]
        ma = x.max(dim=1)[0]
        ss = x.std(dim=1)
        z = torch.hstack((mm, mi, ma, ss))
        out = self.lin(z)
        return out

class Atoy(nn.Module):
    def __init__(self, nh_e, nh_y):
        """ Map edge features to global features.
        """
        super().__init__()
        self.lin = nn.Linear(4 * nh_e, nh_y)

    def forward(self, a):
        """
        a: (bs, n, n, nh_a)
        """
        mm = a.mean(dim=(1, 2))
        mi = a.min(dim=2)[0].min(dim=1)[0]
        ma = a.max(dim=2)[0].max(dim=1)[0]
        ss = torch.std(a, dim=(1, 2))
        z = torch.hstack((mm, mi, ma, ss))
        out = self.lin(z)
        return out


class NodeEdgeAttBlock(nn.Module):
    def __init__(self,
                 nh_n: int,
                 nh_e: int,
                 nh_y: int,
                 n_head: int
                 ):
        super().__init__()
        assert nh_n % n_head == 0, f"nh_n: {nh_n} -- nhead: {n_head}"
        self.nh_n = nh_n
        self.nh_e = nh_e
        self.nh_y = nh_y
        self.nh_f = int(nh_n / n_head)
        self.n_head = n_head

        self.q = Linear(nh_n, nh_n)
        self.k = Linear(nh_n, nh_n)
        self.v = Linear(nh_n, nh_n)

        # FiLM a to x
        self.a2x_add = Linear(nh_e, nh_n)
        self.a2x_mul = Linear(nh_e, nh_n)
        # FiLM y to a
        self.y2e_mul = Linear(nh_y, nh_n)
        self.y2e_add = Linear(nh_y, nh_n)
        # FiLM y to x
        self.y2x_mul = Linear(nh_y, nh_n)
        self.y2x_add = Linear(nh_y, nh_n)

        # Process y
        self.y_y = Linear(nh_y, nh_y)
        self.x_y = Xtoy(nh_n, nh_y)
        self.a_y = Atoy(nh_e, nh_y)

        # Output layers
        self.x_out = Linear(nh_n, nh_n)
        self.a_out = Linear(nh_n, nh_e)
        self.y_out = nn.Sequential(nn.Linear(nh_y, nh_y), nn.ReLU(), nn.Linear(nh_y, nh_y))

    def forward(self, x, a, y):
        """
        x: (bs, n, nh_n)    node features
        a: (bs, n, n, nh_e) edge features
        y: (bs, nh_y)       global features
        Output: xhat, ahat, yhat with the same shape.
        """
        Q = self.q(x)                                   # (bs, n, nh_n)
        K = self.k(x)                                   # (bs, n, nh_n)

        # Reshape to (bs, n, n_head, nh_f) with nh_n = n_head * nh_f
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.nh_f))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.nh_f))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, nh_f)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, nh_f)

        # Compute unnormalized attentions
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))

        # FiLM: Incorporate edge features to the self attention scores
        a1 = self.a2x_add(a)                            # (bs, n, n, nh_n)
        a2 = self.a2x_mul(a)                            # (bs, n, n, nh_n)
        a1 = a1.reshape((a.size(0), a.size(1), a.size(2), self.n_head, self.nh_f))
        a2 = a2.reshape((a.size(0), a.size(1), a.size(2), self.n_head, self.nh_f))
        Y = Y * (a1 + 1) + a2                           # (bs, n, n, n_head, nh_f)

        # FiLM: Incorporate y to a
        ahat = Y.flatten(start_dim=3)                   # (bs, n, n, nh_n)
        ye1 = self.y2e_add(y).unsqueeze(1).unsqueeze(1) # (bs, 1, 1, nh_e)
        ye2 = self.y2e_mul(y).unsqueeze(1).unsqueeze(1)
        ahat = ye1 + (ye2 + 1) * ahat

        # Output a
        ahat = self.a_out(ahat)                         # (bs, n, n, nh_e)

        # Compute attentions.
        attn = torch.softmax(Y, dim=2)                  # (bs, n, n, n_head) ? (bs, n, n, n_head, nh_f)

        V = self.v(x)                                   # (bs, n, nh_n)
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.nh_f))
        V = V.unsqueeze(1)                              # (bs, 1, n, n_head, nh_f)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)    # (bs, n, nh_n)

        # FiLM: Incorporate y to x
        yx1 = self.y2x_add(y).unsqueeze(1)
        yx2 = self.y2x_mul(y).unsqueeze(1)
        xhat = yx1 + (yx2 + 1) * weighted_V

        # Output x
        xhat = self.x_out(xhat)

        # Process y based on x and a
        yhat = self.y_y(y) + self.a_y(a) + self.x_y(x)
        # Output y
        yhat = self.y_out(yhat)                         # (bs, nh_y)

        return xhat, ahat, yhat

class TransformerLayer(nn.Module):
    """ Transformer that updates node and edge features
        nh_n: node features
        nh_e: edge features
        nh_y" global features
        n_head: the number of heads in the multi_head_attention
        df_n: the dimension of the feedforward network for node features
        df_e: the dimension of the feedforward network for edge features
        df_y: the dimension of the feedforward network for global features
        dropout: dropout probablility
        eps: eps value in layer normalizations
    """
    def __init__(self,
                 n_head: int,
                 nh_n: int,
                 nh_e: int,
                 nh_y: int,
                 df_n: int = 2048,
                 df_e: int = 128,
                 df_y: int = 128,
                 dropout: float = 0.1,
                 eps: float = 1e-5
                 ):
        super().__init__()
        self.self_attn = NodeEdgeAttBlock(nh_n, nh_e, nh_y, n_head)

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

        self.lin_y1 = Linear(nh_y, df_y)
        self.lin_y2 = Linear(df_y, nh_y)
        self.norm_y1 = LayerNorm(nh_y, eps=eps)
        self.norm_y2 = LayerNorm(nh_y, eps=eps)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, x: Tensor, a: Tensor, y):
        """
        x: (bs, n, nh_n)
        a: (bs, n, n, nh_e)
        y: (bs, nh_y)
        Output: xhat, ahat, yhat with the same dims.
        """
        xhat, ahat, yhat = self.self_attn(x, a, y)

        x = self.norm_x1(x + self.dropout_x1(xhat))
        a = self.norm_a1(a + self.dropout_a1(ahat))
        y = self.norm_y1(y + self.dropout_y1(yhat))

        ff_output_x = self.lin_x2(self.dropout_x2(self.activation(self.lin_x1(x))))
        ff_output_a = self.lin_a2(self.dropout_a2(self.activation(self.lin_a1(a))))
        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))

        x = self.norm_x2(x + self.dropout_x3(ff_output_x))
        a = self.norm_a2(a + self.dropout_a3(ff_output_a))
        y = self.norm_y2(y + self.dropout_y3(ff_output_y))

        return x, a, y

class GraphTransformer(nn.Module):
    def __init__(self,
                 n_layers: int,
                 n_xi: int,
                 n_ai: int,
                 n_yi: int,
                 n_xo: int,
                 n_ao: int,
                 n_yo: int,
                 mh_n: int,
                 mh_e: int,
                 mh_y: int,
                 n_head: int,
                 nh_n: int,
                 nh_e: int,
                 nh_y: int,
                 df_n: int,
                 df_e: int,
                 df_y: int,
                 act_fn_in = nn.ReLU,
                 act_fn_out = nn.ReLU,
                 device: Optional[str]='cuda'
                 ):
        super().__init__()
        self.n_layers = n_layers
        self.n_xi = n_xi
        self.n_ai = n_ai
        self.n_yi = n_yi
        self.n_xo = n_xo
        self.n_ao = n_ao
        self.n_yo = n_yo

        self.mlp_in_x = nn.Sequential(
            nn.Linear(n_xi, mh_n),
            act_fn_in(),
            nn.Linear(mh_n, nh_n),
            act_fn_in()
        )
        self.mlp_in_a = nn.Sequential(
            nn.Linear(n_ai, mh_e),
            act_fn_in(),
            nn.Linear(mh_e, nh_e),
            act_fn_in()
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(n_yi, mh_y),
            act_fn_in(),
            nn.Linear(mh_y, nh_y),
            act_fn_in()
        )

        self.tf_layers = nn.ModuleList([
            TransformerLayer(
                n_head=n_head,
                nh_n=nh_n,
                nh_e=nh_e,
                nh_y=nh_y,
                df_n=df_n,
                df_e=df_e,
                df_y=df_y
            ) for _ in range(n_layers)
        ])

        self.mlp_out_x = nn.Sequential(
            nn.Linear(nh_n, mh_n),
            act_fn_out(),
            nn.Linear(mh_n, n_xo)
        )
        self.mlp_out_a = nn.Sequential(
            nn.Linear(nh_e, mh_e),
            act_fn_out(),
            nn.Linear(mh_e, n_ao)
        )
        self.mlp_out_y = nn.Sequential(
            nn.Linear(nh_y, mh_y),
            act_fn_out(),
            nn.Linear(mh_y, n_yo)
        )

        self.device = device

    def forward(self, x, a, y):
        # bs, n = x.shape[0], x.shape[1]

        # diag_mask = torch.eye(n)
        # diag_mask = ~diag_mask.type_as(a).bool()
        # diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        # x_to_out = x[..., :self.n_xo]
        # a_to_out = a[..., :self.n_ao]
        # y_to_out = y[..., :self.n_yo]

        x = self.mlp_in_x(x)
        a = self.mlp_in_a(a)
        y = self.mlp_in_y(y)
        a = (a + a.transpose(1, 2)) / 2

        for layer in self.tf_layers:
            x, a, y = layer(x, a, y)

        x = self.mlp_out_x(x)
        a = self.mlp_out_a(a)
        y = self.mlp_out_y(y)

        a = set_diagonal(a, self.device, 0.)
        a = (a + a.transpose(1, 2)) / 2

        # r = x.size(-1)//x_to_out.size(-1)
        # x = (x + x_to_out.repeat(1, 1, r))
        # a = (a + a_to_out.repeat(1, 1, 1, r)) * diag_mask
        # y = (y + y_to_out.repeat(1, r))
        # a = (a + a.transpose(1, 2)) / 2

        return x, a, y


if __name__ == '__main__':
    n_layers = 2
    n_xi = 5
    n_ai = 4
    n_yi = 10
    mh_n = 256
    mh_e = 128
    mh_y = 128
    n_head = 8
    nh_n = 256
    nh_e = 64
    nh_y = 64
    df_n = 256
    df_e = 128
    df_y = 128

    model = GraphTransformer(n_layers, n_xi, n_ai, n_yi, n_xi, n_ai, n_yi, mh_n, mh_e, mh_y, n_head, nh_n, nh_e, nh_y, df_n, df_e, df_y)

    xx = torch.randn(100, 9, n_xi)
    aa = torch.randn(100, 9, 9, n_ai)
    yy = torch.randn(100, n_yi)

    x, a, y = model(xx, aa, yy)
    print(x)
    print(a)
    print(y)

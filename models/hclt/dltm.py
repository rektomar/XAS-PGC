from torch.distributions import Bernoulli, Binomial, Categorical, Dirichlet, Normal
from typing import Union, Optional, List
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class LayerIdx:
    leaf_idx: torch.LongTensor
    sum_idx: torch.LongTensor
    prod_idx: torch.LongTensor


def safelog(x: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    if eps is None:
        eps = torch.finfo(torch.get_default_dtype()).tiny
    return torch.log(torch.clamp(x, min=eps))


def batched_logsumexp(log_prob: torch.Tensor, sum_param: torch.Tensor) -> torch.Tensor:
    # credits to github.com/loreloc
    # log_prob  (batch_size, num_features, hidden_dim), this is in log  domain
    # sum_param (num_features, hidden_dim, hidden_dim), this is in prob domain
    max_log_prob = torch.max(log_prob, dim=-1, keepdim=True).values
    norm_exp_log_prob = torch.exp(log_prob - max_log_prob)
    log_prob = max_log_prob + safelog(norm_exp_log_prob.transpose(0, 1) @ sum_param.transpose(1, 2)).transpose(0, 1)
    return log_prob


class DLTM(torch.nn.Module):
    def __init__(
        self,
        tree: Union[List, np.array],
        leaf_type: str,
        hidden_dim: Optional[int] = 16,
        root_hidden_dim: Optional[int] = 16,
        num_categories: Optional[int] = None,
        norm_weight: Optional[bool] = True,
        learnable: Optional[bool] = True,
        min_std: Optional[float] = 1e-3,
        max_std: Optional[float] = 7.0,
    ):
        super().__init__()
        self.tree = np.array(tree)  # List of predecessors: tree[i] = j if j is parent of i
        self.root = np.argwhere(self.tree == -1).item()  # tree[i] = -1 if i is root
        self.num_features = num_features = len(self.tree)
        self.features = list(range(num_features))
        self.root_hidden_dim = root_hidden_dim
        self.num_categories = num_categories
        self.norm_weight = norm_weight
        self.hidden_dim = hidden_dim
        self.leaf_type = leaf_type
        self.min_std = min_std
        self.max_std = max_std
        self._build_structure()

        self.marginalization_mask = None

        sum_logits = Dirichlet(torch.ones(hidden_dim)).sample([num_features, hidden_dim]).log()
        self.sum_logits = torch.nn.Parameter(sum_logits) if learnable else sum_logits

        match self.leaf_type:
            case 'bernoulli':
                leaf_logits = torch.rand(num_features, hidden_dim)
                leaf_logits.logit_()
            case 'binomial':
                leaf_logits = torch.rand(num_features, hidden_dim)
            case 'categorical':
                leaf_logits = torch.randn(num_features, hidden_dim, num_categories)
            case 'gaussian':
                leaf_logits = torch.randn(num_features, 2, hidden_dim)*0.05
            case _:
                raise NotImplementedError('leaf_type not implemented')

        self.leaf_logits = torch.nn.Parameter(leaf_logits) if learnable else leaf_logits

    def _build_structure(self):
        self.bfs = {0: [[self.root], [-1]]}
        depths = np.array([0 if node == self.root else None for node in range(self.num_features)])
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            children = np.argwhere(self.tree == node)[:, 0].tolist()
            depths[children] = depths[node] + 1
            if len(children):
                queue.extend(children)
                self.bfs.setdefault(depths[node] + 1, [[], []])[0].extend(children)
                self.bfs[depths[node] + 1][1].extend([node] * len(children))

        self.post_order = {}
        last_node_visited = None
        layer_idx = np.full(self.num_features, None)
        stack = [self.root]
        while stack:
            children = list(np.argwhere(self.tree == stack[-1])[:, 0])
            if len(children) == 0 or last_node_visited in children:
                layer_idx[stack[-1]] = 0 if len(children) == 0 else 1 + max(layer_idx[children])
                self.post_order.setdefault(layer_idx[stack[-1]], {})[stack[-1]] = children
                last_node_visited = stack.pop()
            else:
                stack.extend(children)

        self.layers = []
        for layer_id in self.post_order:
            in_len = torch.LongTensor(list(map(len, self.post_order[layer_id].values())))
            self.layers.append(
                LayerIdx(
                    leaf_idx = torch.LongTensor(list(self.post_order[layer_id].keys())),
                    sum_idx  = torch.LongTensor(sum(self.post_order[layer_id].values(), [])),
                    prod_idx = torch.arange(len(in_len)).repeat_interleave(in_len)
                    )
                )

    @property
    def leaf_param(self):
        match self.leaf_type:
            case 'bernoulli':
                return self.leaf_logits.sigmoid()
            case 'binomial':
                return self.leaf_logits.sigmoid()
            case 'categorical':
                return self.leaf_logits.softmax(dim=2)
            case 'gaussian':
                scale = (torch.nn.functional.silu(self.leaf_logits[:, 1:]) + 0.279).clamp(min=self.min_std, max=self.max_std)
                return torch.cat([self.leaf_logits[:, :1], scale], dim=1)
            case _:
                raise NotImplementedError('leaf_type not implemented')

    @property
    def sum_param(self):
        return self.sum_logits.softmax(dim=-1) if self.norm_weight else self.sum_logits

    @property
    def log_norm_constant(self):
        x = torch.zeros(1, self.num_features).to(device=self.sum_logits.device)
        return self.forward(x, has_nan=x == 0).squeeze()

    def set_marginalization_mask(self, mask):
        self.marginalization_mask = mask

    def leaf_log_prob(self, x: torch.Tensor):
        assert x.ndim == 2 and x.size(1) == self.num_features
        leaf_param = self.leaf_param

        match self.leaf_type:
            case 'bernoulli':
                leaf_log_prob = Bernoulli(leaf_param, validate_args=False).log_prob(x.unsqueeze(2))
            case 'binomial':
                leaf_log_prob = Binomial(self.num_categories - 1, leaf_param, validate_args=False).log_prob(x.unsqueeze(2))
            case 'categorical':
                index = x if x.dtype == torch.long else x.long()
                index = index.clamp(0)
                leaf_log_prob = leaf_param.log().transpose(1, 2)[range(self.num_features), index]
            case 'gaussian':
                leaf_log_prob = Normal(leaf_param[:, 0], leaf_param[:, 1], validate_args=False).log_prob(x.unsqueeze(2))
            case _:
                raise NotImplementedError('leaf_type not implemented.')

        if self.marginalization_mask is not None:
            leaf_log_prob.masked_fill(~self.marginalization_mask.unsqueeze(2), 0)

        return leaf_log_prob

    def forward(
        self,
        x: torch.Tensor,
        normalize: Optional[bool] = False,
        return_lls: Optional[bool] = False,
        return_prod_lls: Optional[bool] = False
    ):
        leaf_log_prob = self.leaf_log_prob(x)  # (batch_size, num_features, hidden_dim)
        lls = {'leaf': leaf_log_prob, 'sum': torch.zeros_like(leaf_log_prob)}
        if return_prod_lls: lls['prod'] = torch.zeros_like(leaf_log_prob)
        sum_param = self.sum_param
        for layer in self.layers:
            prod = torch.index_add(
                source=lls['sum'][:, layer.sum_idx], dim=1, index=layer.prod_idx.to(x.device),
                input=lls['leaf'][:, layer.leaf_idx])
            lls['sum'][:, layer.leaf_idx] = batched_logsumexp(prod, sum_param[layer.leaf_idx])
            if return_prod_lls: lls['prod'][:, layer.leaf_idx] = prod
        root_log_prob = lls['sum'][:, self.layers[-1].leaf_idx, 0:self.root_hidden_dim] - (self.log_norm_constant if normalize else 0)
        root_log_prob = root_log_prob.squeeze()
        return (root_log_prob, lls) if return_lls else root_log_prob

    @torch.no_grad()
    def backward(
        self,
        num_samples: Optional[int] = None,
        x: Optional[torch.Tensor] = None,
        class_idxs: Optional[Union[int, torch.Tensor]] = 0,
        mpe: Optional[bool] = False,
        mpe_leaf: Optional[bool] = False
    ):
        def sample_or_mode(dist: torch.distributions.Distribution, mode: bool):
            return dist.mode if mode else dist.sample()

        if x is not None:
            prod_prob = self.forward(x, return_lls=True, return_prod_lls=True)[1]['prod'].exp() # (num_samples, num_features, hidden_dim)
        else:
            prod_prob = torch.ones(num_samples, self.num_features, self.hidden_dim, device=self.sum_logits.device)

        sum_param = self.sum_param
        sum_states = torch.full((len(prod_prob), self.num_features), -1, device=self.sum_logits.device, dtype=torch.long)
        sum_states[:, self.bfs[0][0]] = sample_or_mode(Categorical(probs=sum_param[self.bfs[0][0], class_idxs].unsqueeze(1) * prod_prob[:, self.bfs[0][0]]), mode=mpe)
        for depth in range(1, len(self.bfs)):
            children, parents = self.bfs[depth]
            sum_states[:, children] = sample_or_mode(Categorical(probs=sum_param[children, sum_states[:, parents]] * prod_prob[:, children]), mode=mpe)

        match self.leaf_type:
            case 'bernoulli':
                samples = sample_or_mode(Bernoulli(self.leaf_param[self.features, sum_states]), mode=mpe or mpe_leaf)
            case 'categorical':
                samples = sample_or_mode(Categorical(self.leaf_param[self.features, sum_states]), mode=mpe or mpe_leaf)
            case 'gaussian':
                loc, scale = self.leaf_param.chunk(2, dim=1)
                samples = sample_or_mode(Normal(loc[self.features, 0, sum_states], scale[self.features, 0, sum_states]), mode=mpe or mpe_leaf)
            case _:
                raise NotImplementedError('leaf_type not implemented.')

        if x is not None:
            samples[self.marginalization_mask] = x[self.marginalization_mask].type_as(samples)

        return samples

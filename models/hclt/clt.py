from scipy import sparse as sp
from typing import Optional
import networkx as nx
import torch


def maximum_spanning_tree(root: int, adj_matrix: torch.Tensor):
    mst = sp.csgraph.minimum_spanning_tree(-(adj_matrix.cpu().numpy() + 1.0), overwrite=True)
    bfs, tree = sp.csgraph.breadth_first_order(mst, directed=False, i_start=root, return_predecessors=True)
    tree[root] = -1
    return bfs, tree


def categorical_mutual_info(
    data: torch.LongTensor,
    alpha: float = 0.01,
    num_categories: Optional[int] = None,
    chunk_size: Optional[int] = None
):
    assert data.dtype == torch.long and data.ndim == 2
    num_samples, num_features = data.size()
    if num_categories is None:
        num_categories = int(data.max().item() + 1)
    if chunk_size is None:
        chunk_size = num_samples

    idx_features = torch.arange(0, num_features)
    idx_categories = torch.arange(0, num_categories)

    joint_counts = torch.zeros(num_features, num_features, num_categories ** 2, dtype=torch.long, device=data.device)
    for _, chunk in enumerate(data.split(chunk_size)):
        joint_values = chunk.t().unsqueeze(1) * num_categories + chunk.t().unsqueeze(0)
        joint_counts.scatter_add_(-1, joint_values.long(), torch.ones_like(joint_values))
    joint_counts = joint_counts.view(num_features, num_features, num_categories, num_categories)
    marginal_counts = joint_counts[idx_features, idx_features][:, idx_categories, idx_categories]

    marginals = (marginal_counts + num_categories * alpha) / (num_samples + num_categories ** 2 * alpha)
    joints = (joint_counts + alpha) / (num_samples + num_categories ** 2 * alpha)
    joints[idx_features, idx_features] = torch.diag_embed(marginals)
    outers = torch.einsum('ik, jl -> ijkl', marginals, marginals)

    return (joints * (joints.log() - outers.log())).sum(dim=(2, 3)).fill_diagonal_(0)

def learn_clt(
    data: torch.Tensor,
    leaf_type: str,
    chunk_size: Optional[int] = None,
    num_bins: Optional[int] = None,
    num_categories: Optional[int] = None
):
    if leaf_type in ['bernoulli', 'categorical']:
        if num_bins is not None:
            assert num_categories is not None, 'Number of categories must be known if rescaling in bins'
            data = torch.div(data, num_categories // num_bins, rounding_mode='floor')
        mutual_info = categorical_mutual_info(data.long(), num_categories=num_categories, chunk_size=chunk_size)
    elif leaf_type == 'gaussian':
        mutual_info = (- 0.5 * torch.log(1 - torch.corrcoef(data.t()) ** 2)).numpy()
    else:
        raise NotImplementedError('MI computation not implemented for %s leaves.' % leaf_type)

    _, tree = maximum_spanning_tree(root=0, adj_matrix=mutual_info)
    nx_tree = nx.Graph([(node, parent) for node, parent in enumerate(tree) if parent != -1])
    _, tree = maximum_spanning_tree(root=nx.center(nx_tree)[0], adj_matrix=mutual_info)

    return tree

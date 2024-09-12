import torch
from models.utils import zero_diagonal

# The following code is based on https://github.com/cvignac/DiGress

class ExtraFeatures:
    def __init__(self,
                 type: str,
                 max_nodes: int
                 ):
        self.max_nodes = max_nodes
        self.ncycles = NodeCycleFeatures()
        self.type = type

        if self.type == 'none':
            self.nf_x = 0
            self.nf_a = 0
            self.nf_y = 0
        elif self.type == 'cycles':
            self.nf_x = 3
            self.nf_a = 0
            self.nf_y = 5
        elif self.type == 'eigen':
            self.nf_x = 3
            self.nf_a = 0
            self.nf_y = 11
            self.eigenfeatures = EigenFeatures(type)
        elif self.type == 'all':
            self.nf_x = 6
            self.nf_a = 0
            self.nf_y = 11
            self.eigenfeatures = EigenFeatures(type)
        else:
            raise ValueError(f"Features type {self.type} not implemented")

    def __call__(self, x, a):
        rel_num_nodes = torch.sum(~x[..., -1].bool(), dim=1, keepdim=True) / self.max_nodes
        x_cycles, y_cycles = self.ncycles(a) # (bs, n, 3), (bs, 4)

        if self.type == 'none':
            return (torch.zeros((*x.shape[:-1], 0)).type_as(x), # (bs, n, 0)
                    torch.zeros((*a.shape[:-1], 0)).type_as(a), # (bs, n, n, 0)
                    torch.zeros(( x.shape[0],   0)).type_as(x)  # (bs, 0)
                    )
        elif self.type == 'cycles':
            return (x_cycles,                                   # (bs, n, 3)
                    torch.zeros((*a.shape[:-1], 0)).type_as(a), # (bs, n, n, 0)
                    torch.hstack((rel_num_nodes,
                                  y_cycles))                    # (bs, 1), (bs, 4)
                    )

        elif self.type == 'eigen':
            num_components, batched_eigenvalues = self.eigenfeatures(x, a)

            return (x_cycles,                                   # (bs, n, 3)
                    torch.zeros((*a.shape[:-1], 0)).type_as(a), # (bs, n, n, 0)
                    torch.hstack((rel_num_nodes,
                                  y_cycles,
                                  num_components,
                                  batched_eigenvalues))         # (bs, 1), (bs, 4), (bs, 1), (bs, 5)
                    )

        elif self.type == 'all':
            num_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = self.eigenfeatures(x, a)

            return (torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1), # (bs, n, 3), (bs, n, 1), (bs, n, 2)
                    torch.zeros((*a.shape[:-1], 0)).type_as(a),                       # (bs, n, n, 0)
                    torch.hstack((rel_num_nodes,
                                  y_cycles,
                                  num_components,
                                  batched_eigenvalues))                               # (bs, 1), (bs, 4), (bs, 1), (bs, 5)
                    )
        else:
            raise ValueError(f"Features type {self.type} not implemented")


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, a):
        adj_matrix = a[..., :-1].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)
        x_cycles = x_cycles.type_as(adj_matrix)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from: https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self, type):
        self.type = type

    def __call__(self, x, a):
        mask = ~x[..., -1].bool()
        A = a[..., :-1].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.type == 'eigen':
            eigvals = torch.linalg.eigvalsh(L)        # (bs, n)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            num_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return num_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.type == 'all':
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            num_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                               node_mask=mask,
                                                                               num_connected=num_connected_comp)
            return num_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"The type {self.type} is not implemented")


def laplacian(a, normalize: bool):
    """
    a : adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(a, dim=-1)                          # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - a                                # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    D_norm = torch.diag_embed(1 / torch.sqrt(diag))      # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ a @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: number of non zero eigenvalues to keep
    """
    bs, n = eigenvalues.shape
    num_connected_components = (eigenvalues < 1e-5).sum(dim=-1)
    assert (num_connected_components > 0).all(), (num_connected_components, eigenvalues)

    to_extend = max(num_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + num_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return num_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, num_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask                    # (bs, n)
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)                 # (bs, n)
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values                                    # values: (bs, )
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(num_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)                # (bs, n , n + to_extend)
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + num_connected.unsqueeze(2) # (bs, 1, k)
    indices = indices.expand(-1, n, -1)                                               # (bs, n, k)
    first_k_ev = torch.gather(vectors, dim=2, index=indices)                          # (bs, n, k)
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev

def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy

if __name__ == '__main__':
    max_nodes = 7
    nk_ni = 5
    nk_ei = 4
    device = 'cpu'

    logits_x = torch.randn(100, max_nodes, nk_ni)
    logits_a = torch.randn(100, max_nodes, max_nodes, nk_ei)

    x = torch.distributions.Categorical(logits=logits_x).sample()
    a = torch.distributions.Categorical(logits=logits_a).sample()

    x = torch.nn.functional.one_hot(x.to(torch.int64), num_classes=nk_ni)
    a = torch.nn.functional.one_hot(a.to(torch.int64), num_classes=nk_ei)

    a = zero_diagonal(a, device)
    a = (a + a.transpose(1, 2)) / 2

    # features = DummyExtraFeatures()
    # features = ExtraFeatures('cycles', max_nodes)
    # features = ExtraFeatures('eigen', max_nodes)
    features = ExtraFeatures('all', max_nodes)

    x, a, y = features(x, a)

    print(x)
    print(a)
    print(y)

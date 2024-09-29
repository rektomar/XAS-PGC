import torch

def permute_graph(xx, aa, pi):
    px = xx[pi, ...]
    pa = aa[pi, :, ...]
    pa = pa[:, pi, ...]
    return px, pa

def flatten_graph(xx, aa, dim=2):
    n = xx.shape[1]
    z = torch.cat((xx.unsqueeze(dim), aa), dim=dim)
    return z.view(-1, n + n**2)

def unflatt_graph(z, nd_nodes, num_full):
    z = z.view(-1, nd_nodes, nd_nodes+1)
    x = z[:, 0:num_full, 0 ]
    a = z[:, 0:num_full, 1:num_full+1]
    return x, a

def bandwidth(a, n):
    bands = torch.arange(a.shape[0]) - a.argmax(dim=1)
    return bands[:n].max()

def flatten(a, bandwidth):
    n = a.shape[0]
    a_flat = torch.zeros((n, bandwidth) + a.shape[2:]).type_as(a)
    for i in range(n):
        m = min(bandwidth, n - i - 1)
        if m > 0:
            a_flat[i, :m] = a[i, i+1:i+1+m]
    return a_flat

def unflatten(a_flat):
    n, bandwidth = a_flat.shape[:2]
    a = torch.zeros((n, n) + a_flat.shape[2:]).type_as(a_flat) + 3
    for i in range(n):
        m = min(bandwidth, n - i - 1)
        if m > 0:
            a[i, i+1:i+1+m] = a_flat[i, :m]
            a[i+1:i+1+m, i] = a_flat[i, :m]
    return a

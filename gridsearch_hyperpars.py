import json
import itertools

from utils.templates_hyperpars import *


def grid_btree(
        nl = [2, 3],
        ns = [20, 40],
        ni = [20, 40]
):
    return [template_btree(*p) for p in list(itertools.product(nl, ns, ni))]

def grid_vtree(
        nl = [2, 3],
        ns = [20, 40],
        ni = [20, 40]
):
    return [template_vtree(*p) for p in list(itertools.product(nl, ns, ni))]

def grid_rtree(
        nl = [2, 3],
        nr = [1, 10],
        ns = [20, 40],
        ni = [20, 40]
):
    return [template_rtree(*p) for p in list(itertools.product(nl, nr, ns, ni))]

def grid_ptree(
        nl = [2, 3],
        ns = [20, 40],
        ni = [20, 40]
):
    return [template_ptree(*p) for p in list(itertools.product(nl, ns, ni))]

def grid_ctree(
        nh = [2, 3]
):
    return [template_ctree(p) for p in nh]


def grid_zero_sort(dataset):
    order = ['canonical', 'bft', 'dft', 'rcm', 'unordered']
    nc = [1, 8, 16]
    backend_name = ['btree', 'vtree', 'rtree', 'ptree', 'ctree']
    backend_grid = [grid_btree, grid_vtree, grid_rtree, grid_ptree, grid_ctree]
    backend_xpar = [
        {"nl":[3],           "ns":[40], "ni":[40]},
        {"nl":[3],           "ns":[40], "ni":[40]},
        {"nl":[3], "nr":[1], "ns":[40], "ni":[40]},
        {"nl":[3],           "ns":[40], "ni":[40]},
        {"nh":[64]}
    ]
    backend_apar = [
        {"nl":[5],           "ns":[40], "ni":[40]},
        {"nl":[5],           "ns":[40], "ni":[40]},
        {"nl":[5], "nr":[1], "ns":[40], "ni":[40]},
        {"nl":[5],           "ns":[40], "ni":[40]},
        {"nh":[64]}
    ]
    backend_nr = [
        [None],
        [None],
        [None],
        [1, 10],
        [None]
    ]
    batch_size = [256]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    hyperpars = []
    for b_name, b_grid, b_xpar, b_apar, b_nr in zip(backend_name, backend_grid, backend_xpar, backend_apar, backend_nr):
        grid = itertools.product(order, nc, b_nr, [b_name], b_grid(**b_xpar), b_grid(**b_apar), batch_size, lr, seed)
        hyperpars.extend([template_zero_sort(dataset, *p) for p in list(grid)])

    return hyperpars


GRIDS = {
    'zero_sort': grid_zero_sort,
}


if __name__ == "__main__":
    print(len(grid_zero_sort('qm9')))
    # for p in grid_zero_sort('qm9')[-9:]:
    #     print(json.dumps(p, indent=4))

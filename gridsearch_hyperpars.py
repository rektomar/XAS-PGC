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


def grid_sort(dataset, model):
    order = ['canonical', 'bft', 'dft', 'rcm', 'unordered']
    nc = [16]
    backend_name = ['btree', 'vtree', 'rtree', 'ptree', 'ctree']
    backend_grid = [grid_btree, grid_vtree, grid_rtree, grid_ptree, grid_ctree]
    match dataset:
        case 'qm9':
            backend_xpar = [
                {"nl":[3],            "ns":[32], "ni":[32]},
                {"nl":[3],            "ns":[32], "ni":[32]},
                {"nl":[3], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[3],            "ns":[32], "ni":[32]},
                {"nh":[256]}
            ]
            backend_apar = [
                {"nl":[5],            "ns":[32], "ni":[32]},
                {"nl":[5],            "ns":[32], "ni":[32]},
                {"nl":[5], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[5],            "ns":[32], "ni":[32]},
                {"nh":[256]}
            ]
        case 'zinc250k':
            backend_xpar = [
                {"nl":[4],            "ns":[32], "ni":[32]},
                {"nl":[4],            "ns":[32], "ni":[32]},
                {"nl":[4], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[4],            "ns":[32], "ni":[32]},
                {"nh":[256]}
            ]
            backend_apar = [
                {"nl":[6],            "ns":[32], "ni":[32]},
                {"nl":[6],            "ns":[32], "ni":[32]},
                {"nl":[6], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[6],            "ns":[32], "ni":[32]},
                {"nh":[256]}
            ]
        case _:
            raise 'Unknown dataset'
    backend_nr = [
        [None],
        [None],
        [None],
        [16],
        [None]
    ]
    batch_size = [256]
    lr = [0.05]
    seed = [1]

    hyperpars = []
    for b_name, b_grid, b_xpar, b_apar, b_nr in zip(backend_name, backend_grid, backend_xpar, backend_apar, backend_nr):
        grid = itertools.product(order, nc, b_nr, [b_name], b_grid(**b_xpar), b_grid(**b_apar), batch_size, lr, seed)
        hyperpars.extend([template_sort(dataset, model, *p) for p in list(grid)])

    return hyperpars


GRIDS = {
    # 'zero_sort': grid_sort,
    'marg_sort': grid_sort,
}


if __name__ == "__main__":
    print(len(grid_sort('qm9', 'marg_sort')))
    # for p in grid_sort('qm9', 'marg_sort')[-9:]:
    #     print(json.dumps(p, indent=4))

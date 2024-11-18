import json
import itertools

from utils.templates_hyperpars import *


def grid_backend_btree(
        nl = [2, 3],
        nr = [1],
        ns = [20, 40],
        ni = [20, 40]
):
    grid = itertools.product(nl, nr, ns, ni)

    return [backend_btree(*p) for p in list(grid)]


def grid_zero_sort(dataset):
    order = ['unordered', 'canonical', 'bft', 'dft', 'rcm', 'rand']
    backend_x_name = ['btree', 'btree']
    backend_x_hpar = [
        {"nl":[3, 4], "nr":[1], "ns":[20, 40], "ni":[20, 40]},
        {"nl":[3, 4], "nr":[1], "ns":[20, 40], "ni":[20, 40]}]
    backend_x_grid = [grid_backend_btree, grid_backend_btree]
    backend_a_name = ['btree', 'btree']
    backend_a_hpar = [
        {"nl":[5, 6], "nr":[1], "ns":[20, 40], "ni":[20, 40]},
        {"nl":[5, 6], "nr":[1], "ns":[20, 40], "ni":[20, 40]}]
    backend_a_grid = [grid_backend_btree, grid_backend_btree]
    batch_size = [256]
    lr = [0.05]
    seed = [0, 1, 2, 3, 4]

    hyperpars = []
    for bx_name, bx_grid, bx_hpar in zip(backend_x_name, backend_x_grid, backend_x_hpar):
        for ba_name, ba_grid, ba_hpar in zip(backend_a_name, backend_a_grid, backend_a_hpar):
            grid = itertools.product(order, [bx_name], bx_grid(**bx_hpar), [ba_name], ba_grid(**ba_hpar), batch_size, lr, seed)
            hyperpars.extend([template_zero_sort(dataset, *p) for p in list(grid)])

    return hyperpars


GRIDS = {
    'zero_sort': grid_zero_sort,
}


if __name__ == "__main__":
    print(len(grid_zero_sort('qm9')))
    # for p in grid_zero_sort('qm9'):
    #     print(json.dumps(p, indent=4))

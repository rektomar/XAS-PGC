import os
import itertools
from einsum import Graph, EinsumNetwork, ExponentialFamilyArray


class BTreeSPN(EinsumNetwork.EinsumNetwork):
    def __init__(self,
                 nd,
                 nk,
                 nc,
                 nl,
                 ns,
                 ni
                 ):
        args = EinsumNetwork.Args(
            num_var=nd,
            num_dims=1,
            num_input_distributions=ni,
            num_sums=ns,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk},
            use_em=False)
        graph = Graph.binary_tree(nd, nl)

        super().__init__(graph, args)
        self.initialize()

class RTreeSPN(EinsumNetwork.EinsumNetwork):
    def __init__(self,
                 nd,
                 nk,
                 nc,
                 nl,
                 nr,
                 ns,
                 ni
                 ):
        args = EinsumNetwork.Args(
            num_var=nd,
            num_dims=1,
            num_input_distributions=ni,
            num_sums=ns,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk},
            use_em=False)
        graph = Graph.random_binary_trees(nd, nl, nr)

        super().__init__(graph, args)
        self.initialize()


def backend_selector(x, a, hpars):
    nd_x = x.size(1)
    nd_a = a.size(1)
    nk_x = len(x.unique())
    nk_a = len(a.unique())
    nc = hpars['nc']

    perms = itertools.permutations(range(nd_x))

    match hpars['bx']:
        case 'btree':
            network_x = BTreeSPN(nd_x, nk_x, nc, **hpars['bx_hpars'])
        case 'rtree':
            network_x = RTreeSPN(nd_x, nk_x, nc, **hpars['bx_hpars'])
        case _:
            os.error('Unknown backend_x')

    match hpars['ba']:
        case 'btree':
            network_a = BTreeSPN(nd_a, nk_a, nc, **hpars['ba_hpars'])
        case 'rtree':
            network_a = RTreeSPN(nd_a, nk_a, nc, **hpars['ba_hpars'])
        case _:
            os.error('Unknown backend_a')

    return network_x, nd_x, nk_x, network_a, nd_a, nk_a

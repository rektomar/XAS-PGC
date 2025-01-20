import os
import torch
import itertools

from models.einsum import Graph, EinsumNetwork, ExponentialFamilyArray
from models.hclt.clt import learn_clt
from models.hclt.dltm import DLTM


def permute_tril(nd_x, perms_x):
    nd_a = nd_x * (nd_x - 1) / 2
    m = torch.tril(torch.ones(nd_x, nd_x, dtype=torch.bool), diagonal=-1)
    a = torch.zeros(nd_x, nd_x, dtype=torch.int)
    l = torch.arange(nd_a, dtype=torch.int)
    a[m] = l
    a.transpose(0, 1)[m] = l

    perms_a = []
    for perm in perms_x:
        b = a[perm, :]
        b = b[:, perm]
        perms_a.append(b[m].tolist())

    return perms_a

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
        graph = Graph.binary_tree(nd, nl, 'half')

        super().__init__(graph, args)
        self.initialize()

class VTreeSPN(EinsumNetwork.EinsumNetwork):
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
        graph = Graph.binary_tree(nd, nl, 'first')

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

class PTreeSPN(EinsumNetwork.EinsumNetwork):
    def __init__(self,
                 nd,
                 nk,
                 nc,
                 perms,
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
        graph = Graph.permuted_binary_trees(perms, nl)

        super().__init__(graph, args)
        self.initialize()

class CTreeSPN(DLTM):
    def __init__(self,
                 x,
                 nd,
                 nk,
                 nc,
                 nh
                 ):

        tree_x = learn_clt(x.to('cuda'), 'categorical', 10)
        # tree_x = list(range(1, math.ceil(nd / 2))) + [-1] + list(range(math.ceil(nd / 2)-1, nd-1))

        super().__init__(tree_x, 'categorical', nh, nc, nk)

    def sample(self, num_samples=1, class_idxs=None, x=None):
        return self.backward(num_samples, class_idxs=class_idxs, x=x).to(dtype=torch.float)

    def mpe(self, num_samples=1, class_idxs=None, x=None):
        return self.backward(num_samples, class_idxs=class_idxs, x=x, mpe=True, mpe_leaf=True).to(dtype=torch.float)


def backend_selector(x, a, hpars, nk_x_offset=False):
    nd_x = x.size(1)
    nd_a = a.size(1)
    nk_x = len(x.unique())
    nk_a = len(a.unique())
    nc = hpars['nc']

    if nk_x_offset == True:
        nk_x -= 1
        x -= 1

    match hpars['backend']:
        case 'btree':
            network_x = BTreeSPN(   nd_x, nk_x, nc, **hpars['bx_hpars'])
            network_a = BTreeSPN(   nd_a, nk_a, nc, **hpars['ba_hpars'])
        case 'vtree':
            network_x = VTreeSPN(   nd_x, nk_x, nc, **hpars['bx_hpars'])
            network_a = VTreeSPN(   nd_a, nk_a, nc, **hpars['ba_hpars'])
        case 'rtree':
            network_x = RTreeSPN(   nd_x, nk_x, nc, **hpars['bx_hpars'])
            network_a = RTreeSPN(   nd_a, nk_a, nc, **hpars['ba_hpars'])
        case 'ctree':
            network_x = CTreeSPN(x, nd_x, nk_x, nc, **hpars['bx_hpars'])
            network_a = CTreeSPN(a, nd_a, nk_a, nc, **hpars['ba_hpars'])
        case 'ptree':
            perms_x = list(itertools.islice(itertools.permutations(range(nd_x)), hpars['nr']))
            perms_a = permute_tril(nd_x, perms_x)

            network_x = PTreeSPN(nd_x, nk_x, nc, perms_x, **hpars['bx_hpars'])
            network_a = PTreeSPN(nd_a, nk_a, nc, perms_a, **hpars['ba_hpars'])
        case _:
            os.error('Unknown backend')

    return network_x, nd_x, nk_x, network_a, nd_a, nk_a


if __name__ == '__main__':
    nd_x = 5
    nr = 10
    perms_x = list(itertools.permutations(range(nd_x)))[0:nr]
    perms_a = permute_tril(nd_x, perms_x)

    print(perms_x)
    print(perms_a)

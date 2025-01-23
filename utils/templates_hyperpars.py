import os
import json

from utils.datasets import MOLECULAR_DATASETS

NUM_EPOCHS = 40


def template_btree(nl: int=4,            ns: int=40, ni: int=40): return {"nl": nl,           "ns": ns, "ni": ni}
def template_vtree(nl: int=4,            ns: int=40, ni: int=40): return {"nl": nl,           "ns": ns, "ni": ni}
def template_rtree(nl: int=4, nr: int=1, ns: int=40, ni: int=40): return {"nl": nl, "nr": nr, "ns": ns, "ni": ni}
def template_ptree(nl: int=4,            ns: int=40, ni: int=40): return {"nl": nl,           "ns": ns, "ni": ni}
def template_ctree(nh: int=64): return {"nh": nh}


def template_sort(
    dataset: str,
    model: str,
    order: str = "canonical",
    nc: int = 100,
    nr: int = None,
    backend: str = "btree",
    bx_hpars: dict = template_btree(),
    ba_hpars: dict = template_btree(),
    batch_size: int = 1000,
    lr: float = 0.05,
    seed: int = 0
):
    hpars = {
        "dataset": dataset,
        "order": order,
        "model": model,
        "model_hpars": {
            "nc": nc,
            "backend": backend,
            "bx_hpars": bx_hpars,
            "ba_hpars": ba_hpars,
            "device": "cuda"
        },
        "optimizer": "adam",
        "optimizer_hpars": {
            "lr": lr,
            "betas": [
                0.9,
                0.82
            ]
        },
        "num_epochs": NUM_EPOCHS,
        "batch_size": batch_size,
        "seed": seed
    }

    if nr is not None:
        hpars["model_hpars"]["nr"] = nr

    return hpars


HYPERPARS_TEMPLATES = [
    template_sort,
]


if __name__ == '__main__':
    for dataset in MOLECULAR_DATASETS.keys():
        dir = f'config/{dataset}'
        if os.path.isdir(dir) != True:
            os.makedirs(dir)
        for template in HYPERPARS_TEMPLATES:
            hyperpars = template(dataset)
            with open(f'{dir}/{hyperpars["model"]}.json', 'w') as f:
                json.dump(hyperpars, f, indent=4)

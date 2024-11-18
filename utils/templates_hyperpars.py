import os
import json

from utils.datasets import MOLECULAR_DATASETS

NUM_EPOCHS = 40


def backend_btree(
        nl: int = 4,
        nr: int = 1,
        ns: int = 40,
        ni: int = 40
):
    return {
    "nl": nl,
    "nr": nr,
    "ns": ns,
    "ni": ni
}


def template_zero_sort(
        dataset: str,
        order: str = "canonical",
        bx: str = "btree",
        bx_hpars: dict = backend_btree(),
        ba: str = "btree",
        ba_hpars: dict = backend_btree(),
        batch_size: int = 1000,
        lr: float = 0.05,
        seed: int = 0
):
    return {
    "dataset": dataset,
    "order": order,
    "model": "zero_sort",
    "model_hpars": {
        "nc": 100,
        "bx": bx,
        "bx_hpars": bx_hpars,
        "ba": ba,
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


HYPERPARS_TEMPLATES = [
    template_zero_sort,
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

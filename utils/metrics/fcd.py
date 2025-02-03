import numpy as np
from fcd_torch import FCD


def metric_f(smls_gen, smls_ref, device="cuda", canonical=True):
    if len(smls_gen) < 2 or len(smls_ref) < 2:
        return np.nan
    fcd = FCD(device=device, n_jobs=2, canonize=canonical)
    return fcd(smls_ref, smls_gen)

if __name__ == '__main__':
    smls_1 = [
        'CCC1(C)CN1C(C)=O',
        'O=CC1=COC(=O)N=C1',
        'O=CC1(C=O)CN=CO1',
    ]

    smls_2 = [
        'CC1(C)CN1C(C)=O',
        'O=CC1=COC(=O)N=C1',
        'O=CC1(C=O)CN=CO1'
        ]

    fcd = metric_f(smls_1, smls_2)
    print(f'fcd: {fcd}')
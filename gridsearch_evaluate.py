import os
import numpy as np
import pandas as pd

from pylatex import Document, Package, NoEscape


BACKEND_NAMES = {
    'btree': 'BTree',
    'vtree': 'VTree',
    'rtree': 'RAT',
    'ptree': 'RAT-S',
    'ctree': 'HCLT'
}

IGNORE = [
    'device',
    'seed',
    'sam_valid',
    'sam_unique',
    'sam_unique_abs',
    'sam_novel',
    'sam_novel_abs',
    'sam_score',
    'sam_m_stab',
    'sam_a_stab',
    'sam_fcd_val',
    'sam_kldiv_val',
    'sam_nspdk_val',
    'sam_fcd_tst',
    'sam_kldiv_tst',
    'sam_nspdk_tst',
    'res_valid',
    'res_unique',
    'res_unique_abs',
    'res_novel',
    'res_novel_abs',
    'res_score',
    'res_m_stab',
    'res_a_stab',
    'res_fcd_val',
    'res_kldiv_val',
    'res_nspdk_val',
    'res_fcd_tst',
    'res_kldiv_tst',
    'res_nspdk_tst',
    'cor_valid',
    'cor_unique',
    'cor_unique_abs',
    'cor_novel',
    'cor_novel_abs',
    'cor_score',
    'cor_m_stab',
    'cor_a_stab',
    'cor_fcd_val',
    'cor_kldiv_val',
    'cor_nspdk_val',
    'cor_fcd_tst',
    'cor_kldiv_tst',
    'cor_nspdk_tst',
    'time_sam',
    'time_res',
    'time_cor',
    'num_params',
    ]

COLUMN_NAMES = [
        'Model',       'Valid', 'NSPDK', 'FCD', 'Unique',   'Novel']
def baseline_models_qm9():
    rows = [
        ['GraphAF',      77.43,   0.020,  5.27,    88.64,     86.59],
        ['GraphDF',      93.88,   0.064, 10.93,    98.58,     98.54],
        ['MoFlow',       91.36,   0.017,  4.47,    98.65,     94.72],
        ['EDP-GNN',      47.52,   0.005,  2.68,    99.25,     86.58],
        ['GraphEBM',      8.22,   0.030,  6.14,    97.90,     97.01],
        ['SPECTRE',      87.30,   0.163, 47.96,    35.70,     97.28],
        ['GDSS',         95.72,   0.003,  2.90,    98.46,     86.27],
        ['DiGress',      99.00,   0.005,  0.36,    96.66,     33.40],
        ['GRAPHARM',     90.25,   0.002,  1.22,    95.62,     70.39]
    ]
    return pd.DataFrame(rows, columns=COLUMN_NAMES)
def baseline_models_zinc250k():
    rows = [
        ['GraphAF',      68.47,   0.044, 16.02,    98.64,     100.0],
        ['GraphDF',      90.61,   0.177, 33.55,    99.63,     100.0],
        ['MoFlow',       63.11,   0.046, 20.93,    99.99,     100.0],
        ['EDP-GNN',      82.97,   0.049, 16.74,    99.79,     100.0],
        ['GraphEBM',      5.29,   0.212, 35.47,    98.79,     100.0],
        ['SPECTRE',      90.20,   0.109, 18.44,    67.05,     100.0],
        ['GDSS',         97.01,   0.019, 14.66,    99.64,     100.0],
        ['DiGress',      91.02,   0.082, 23.06,    81.23,     100.0],
        ['GRAPHARM',     88.23,   0.055, 16.26,    99.46,     100.0]
    ]
    return pd.DataFrame(rows, columns=COLUMN_NAMES)

def highlight_top3(x, type='max'):
    styles = np.array(len(x)*[None])
    match type:
        case 'min':
            i = np.argsort(x)[:3]
        case 'max':
            i = np.argsort(x)[-3:][::-1]
        case _:
            os.error('Unsupported type.')

    styles[i] = [
        'color:{c1};textbf:--rwrap;',
        'color:{c2};textbf:--rwrap;',
        'color:{c3};textbf:--rwrap;'
    ]

    return styles

def latexify_style(df, path, row_names=None, column_names=None, precision=2):
    if row_names is not None:
        df.replace(row_names, inplace=True)
    if column_names is not None:
        df.rename(columns={o: n for o, n in zip(df.columns, column_names)}, inplace=True)

    subset_min = [('QM9','NSPDK'), ('QM9','FCD'), ('Zinc250k','NSPDK'), ('Zinc250k','FCD')]
    subset_max = [('QM9','Valid'), ('QM9','Unique'), ('QM9','Novel'), ('Zinc250k','Valid'), ('Zinc250k','Unique'), ('Zinc250k','Novel')]

    s = df.style
    s = s.apply(highlight_top3, type='min', subset=subset_min)
    s = s.apply(highlight_top3, type='max', subset=subset_max)
    s.hide()
    s.format(precision=precision)
    s.format(precision=3, subset=[('QM9','NSPDK'), ('Zinc250k','NSPDK')])
    s.to_latex(path, hrules=True, multicol_align='c')

    with open(path, 'r') as file_data:
        lines = file_data.read().splitlines()
        lines.insert(3, '\\cmidrule(lr){2-6}')
        lines.insert(4, '\\cmidrule(lr){7-11}')
        lines.insert(16, '\\midrule')
        print('\n'.join(lines), file=open(path, 'w'))

def latexify_table(r_name, w_name, clean_tex=True):
    with open(r_name) as f:
        table = f.read()

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '0cm'})
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('xcolor', options='table'))
    doc.packages.append(NoEscape(r'\definecolor{c1}{RGB}{27,158,119}'))
    doc.packages.append(NoEscape(r'\definecolor{c2}{RGB}{117,112,179}'))
    doc.packages.append(NoEscape(r'\definecolor{c3}{RGB}{217,95,2}'))
    doc.append(NoEscape(table))
    doc.generate_pdf(f'{w_name}', clean_tex=clean_tex)


def find_best_backends(evaluation_dir, dataset, model, backends):
    d_frame = pd.DataFrame(0., index=range(len(backends.keys())), columns=COLUMN_NAMES)
    path = evaluation_dir + f'{dataset}/{model}/'

    for i, backend in enumerate(backends.keys()):
        b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path) if backend in f])
        g_frame = b_frame.groupby(list(filter(lambda x: x not in IGNORE, b_frame.columns)))
        a_frame = g_frame.agg({'sam_nspdk_val': 'mean'})
        f_frame = g_frame.get_group(a_frame['sam_nspdk_val'].idxmin())

        d_frame.loc[i] = [BACKEND_NAMES[backend],
                          100*f_frame['sam_valid'].mean(),
                              f_frame['sam_nspdk_tst'].mean(),
                              f_frame['sam_fcd_tst'].mean(),
                          100*f_frame['sam_unique'].mean(),
                          100*f_frame['sam_novel'].mean()]

    return d_frame


if __name__ == "__main__":
    evaluation_dir = 'results/gridsearch/model_evaluation/metrics/'

    baselines_qm9 = baseline_models_qm9()
    ourmodels_qm9 = find_best_backends(evaluation_dir, 'qm9', 'zero_sort', BACKEND_NAMES)
    allmodels_qm9 = pd.concat([baselines_qm9, ourmodels_qm9], ignore_index=True)

    baselines_zinc250k = baseline_models_zinc250k()
    ourmodels_zinc250k = find_best_backends(evaluation_dir, 'zinc250k', 'zero_sort', BACKEND_NAMES)
    allmodels_zinc250k = pd.concat([baselines_zinc250k, ourmodels_zinc250k], ignore_index=True)

    allmodels = allmodels_qm9.merge(allmodels_zinc250k, how='left', on='Model', suffixes=('-x', '-y'))
    allmodels.columns = COLUMN_NAMES + COLUMN_NAMES[1:]

    columns = [('','Models')] + [('QM9', name) for name in COLUMN_NAMES[1:]] + [('Zinc250k', name) for name in COLUMN_NAMES[1:]]
    allmodels.columns = pd.MultiIndex.from_tuples(columns)
    allmodels.head()

    latexify_style(allmodels, 'unconditional.tab')
    latexify_table('unconditional.tab', 'unconditional')

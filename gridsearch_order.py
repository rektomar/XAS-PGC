import os
import pandas as pd

from pylatex import Document, TikZ, NoEscape
from gridsearch_evaluate import IGNORE

ORDER_NAMES = ['bft', 'canonical', 'dft', 'rcm', 'unordered']


def find_best(evaluation_dir, dataset, model):
    path = evaluation_dir + f'{dataset}/{model}/'
    b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path)])
    g_frame = b_frame.groupby(['backend', 'order'])

    f_frame = []
    for df in g_frame:
        df[1].dropna(axis=1, how='all', inplace=True)
        gf = df[1].groupby(list(filter(lambda x: x not in IGNORE, df[1].columns)))
        af = gf.agg({'sam_valid': 'mean'})
        ff = gf.get_group(af['sam_valid'].idxmax())
        f_frame.append(ff[['backend', 'order', 'sam_valid', 'sam_nspdk_tst', 'sam_fcd_tst', 'sam_unique', 'sam_novel']])

    f_frame = pd.concat(f_frame).groupby(['backend', 'order'])
    f_frame_m = f_frame.mean()
    f_frame_s = f_frame.std()

    return f_frame_m, f_frame_s

def nextgrouplot(pic, data, ylabel, args=None):
    ngp = f'\\nextgroupplot[xlabel={{Ordering (-)}}, ylabel={{{ylabel} (-)}}, bar width=4pt,'
    if args is not None:
        ngp += f', {args}]'
    else:
        ngp += r']'

    backends = []
    pic.append(NoEscape(ngp))
    for i, (backend, values) in enumerate(data.groupby(level=0)):
        pic.append(NoEscape(f'\\addplot+[ybar, fill=c{i}] plot coordinates {{' + ' '.join(f'({k}, {v})' for k, v in values.droplevel(0).to_dict().items()) + '};'))
        backends.append(backend)
    pic.append(NoEscape(r'\legend{' + ', '.join(f'\\strut {x}' for x in backends) + r'}'))

if __name__ == "__main__":
    evaluation_dir = 'results/gridsearch/model_evaluation/metrics/'

    model = 'zero_sort'
    dataset = 'qm9'
    ylim_nspdk = 0.1
    ylim_fcd = 10.0

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '1cm'})
    doc.packages.append(NoEscape(r'\usepackage{pgfplots}'))
    doc.packages.append(NoEscape(r'\pgfplotsset{compat=1.18}'))
    doc.packages.append(NoEscape(r'\usepgfplotslibrary{groupplots}'))

    doc.packages.append(NoEscape(r'\definecolor{c0}{RGB}{27,158,119}'))
    doc.packages.append(NoEscape(r'\definecolor{c1}{RGB}{117,112,179}'))
    doc.packages.append(NoEscape(r'\definecolor{c2}{RGB}{217,95,2}'))
    doc.packages.append(NoEscape(r'\definecolor{c3}{RGB}{231,41,138}'))
    doc.packages.append(NoEscape(r'\definecolor{c4}{RGB}{230,171,2}'))
    doc.packages.append(NoEscape(r'\definecolor{c5}{RGB}{166,118,29}'))

    with doc.create(TikZ()) as pic:
        pic.append(NoEscape(r'\pgfplotsset{every tick label/.append style={font=\footnotesize}}'))
        pic.append(NoEscape(
            r'\begin{groupplot}[' +
                r'group style={group size=5 by 1, horizontal sep=55pt, vertical sep=35pt},' +
                r'ybar,' +
                r'xtick=data,' +
                r'enlarge x limits=0.1,' +
                r'height=5cm,' +
                r'width=6.4cm,' +
                r'symbolic x coords={' + ', '.join(x for x in ORDER_NAMES) + r'},' +
                r'ymin=0,' +
                r'ymax=1,' +
                r'legend style={font=\tiny,fill=none,draw=none,row sep=-3pt},' +
                r'legend pos=south west,' +
                r'legend cell align=left,' +
                r'label style={font=\footnotesize},' +
                # r'y label style={at={(-0.12,0.5)}},' +
                # r'x label style={at={(0.5,-0.09)}}' +
            r']'
        ))

        frame_m, frame_s = find_best(evaluation_dir, dataset, model)

        nextgrouplot(pic, frame_m['sam_valid'],     'Valid')
        nextgrouplot(pic, frame_m['sam_unique'],    'Unique')
        nextgrouplot(pic, frame_m['sam_novel'],     'Novel')
        nextgrouplot(pic, frame_m['sam_fcd_tst'],   'FCD',   f'ymax={ylim_fcd}')
        nextgrouplot(pic, frame_m['sam_nspdk_tst'], 'NSPDK', f'ymax={ylim_nspdk}, ' + r'y label style={at={(-0.22,0.5)}}, legend to name=legend')

        pic.append(NoEscape(r'\end{groupplot}'))

    doc.generate_pdf('results/gridsearch_order', clean_tex=False)

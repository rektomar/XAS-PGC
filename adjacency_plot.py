import torch

from utils.datasets import load_dataset, MOLECULAR_DATASETS
from utils.graphs import unflatt_tril
from pylatex import Document, TikZ, NoEscape


def nextgrouplot(pic, matrix, title):
    s = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # s.append(f'({i},{j},{matrix[i, j]})')
            s.append(f'({i+1},{j+1}) [{matrix[i, j]}]')
        s.append('\n\n')
    s = ' '.join(s)

    pic.append(NoEscape(f'\\nextgroupplot[title={title},xlabel={{node (-)}},ylabel={{node (-)}}]'))
    pic.append(NoEscape(f'\\addplot[matrix plot, point meta=explicit] coordinates {{\n' + s + '\n};'))

if __name__ == "__main__":
    dataset = 'zinc250k'
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']

    loader_trn_uno, _ = load_dataset(dataset, 100, split=[0.99, 0.01], order='unordered')
    loader_trn_rnd, _ = load_dataset(dataset, 100, split=[0.99, 0.01], order='rand')
    loader_trn_can, _ = load_dataset(dataset, 100, split=[0.99, 0.01], order='canonical')
    loader_trn_bft, _ = load_dataset(dataset, 100, split=[0.99, 0.01], order='bft')
    loader_trn_dft, _ = load_dataset(dataset, 100, split=[0.99, 0.01], order='dft')
    loader_trn_rcm, _ = load_dataset(dataset, 100, split=[0.99, 0.01], order='rcm')

    a_uno = torch.stack([b['a'] for b in loader_trn_uno.dataset])
    a_rnd = torch.stack([b['a'] for b in loader_trn_rnd.dataset])
    a_can = torch.stack([b['a'] for b in loader_trn_can.dataset])
    a_bft = torch.stack([b['a'] for b in loader_trn_bft.dataset])
    a_dft = torch.stack([b['a'] for b in loader_trn_dft.dataset])
    a_rcm = torch.stack([b['a'] for b in loader_trn_rcm.dataset])

    a_uno = unflatt_tril(a_uno, max_atoms)
    a_rnd = unflatt_tril(a_rnd, max_atoms)
    a_can = unflatt_tril(a_can, max_atoms)
    a_bft = unflatt_tril(a_bft, max_atoms)
    a_dft = unflatt_tril(a_dft, max_atoms)
    a_rcm = unflatt_tril(a_rcm, max_atoms)

    a_uno = (a_uno > 0).to(torch.float).mean(dim=0)
    a_rnd = (a_rnd > 0).to(torch.float).mean(dim=0)
    a_can = (a_can > 0).to(torch.float).mean(dim=0)
    a_bft = (a_bft > 0).to(torch.float).mean(dim=0)
    a_dft = (a_dft > 0).to(torch.float).mean(dim=0)
    a_rcm = (a_rcm > 0).to(torch.float).mean(dim=0)

    # a_uno = (a_uno > 0.).to(torch.float)
    # a_rnd = (a_rnd > 0.).to(torch.float)
    # a_can = (a_can > 0.).to(torch.float)
    # a_bft = (a_bft > 0.).to(torch.float)
    # a_dft = (a_dft > 0.).to(torch.float)
    # a_rcm = (a_rcm > 0.).to(torch.float)

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '1cm'})
    doc.packages.append(NoEscape(r'\usepackage{pgfplots}'))
    doc.packages.append(NoEscape(r'\pgfplotsset{compat=1.18}'))
    doc.packages.append(NoEscape(r'\usepgfplotslibrary{groupplots}'))

    with doc.create(TikZ()) as pic:
        # pic.append(NoEscape(r'\pgfplotsset{every tick label/.append style={font=\footnotesize}}'))
        # pic.append(NoEscape(r'\begin{groupplot}[group style={group size=6 by 1, horizontal sep=35pt},view={0}{90},colormap name=viridis,height=5cm,width=5cm,label style={font=\footnotesize},y label style={at={(0.08,0.5)}},x label style={at={(0.5,0.05)}}]'))
        pic.append(NoEscape(f'\\begin{{groupplot}}[group style={{group size=6 by 1}},height=5cm,width=5cm,xmin=0.5,xmax={max_atoms}.5,ymin=0.5,ymax={max_atoms}.5]'))

        nextgrouplot(pic, a_rnd, 'Unordered')
        nextgrouplot(pic, a_rnd, 'Random')
        nextgrouplot(pic, a_can, 'Canonical')
        nextgrouplot(pic, a_bft, 'BFT')
        nextgrouplot(pic, a_dft, 'DFT')
        nextgrouplot(pic, a_rcm, 'Reverse Cuthill-McKee')

        pic.append(NoEscape(r'\end{groupplot}'))

    doc.generate_pdf('adjacency_plot', clean_tex=False)

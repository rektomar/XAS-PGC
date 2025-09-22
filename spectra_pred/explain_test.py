import os
import torch
import matplotlib.pyplot as plt

from rdkit.Chem import Draw

from pylatex import Document, NoEscape
from pylatex import TikZ, TikZNode, TikZOptions


from utils.datasets import MOLECULAR_DATASETS, load_dataset
from utils.datasets import MIN_E, MAX_E, N_GRID
from utils.explain import choose_range, forward_search, bruteforce_search, g2mol_e, mask_graph


def make_pdf_from_grid(out_file="expl_grid"):
    doc = Document(documentclass="standalone",
                   document_options=("preview",),
                   geometry_options={"margin": "0cm"})
    doc.append(NoEscape(r"\usetikzlibrary{positioning}"))

    m_width = 100  # px width of each image

    with doc.create(TikZ(options=TikZOptions({"node distance": "2px and 2px"}))) as pic:
        pic.append(TikZNode(
                    text=rf"\includegraphics[width={m_width}px]{{{'expl_1/before_explain.png'}}}",
                    handle="n10"))
        pic.append(TikZNode(
                    text=rf"\includegraphics[width={0.8*m_width}px]{{{'expl_1/mol_1.png'}}}",
                    options=TikZOptions({"right": f"of n10"}),
                    handle="n11"))
        pic.append(TikZNode(
                text=rf"\includegraphics[width={m_width}px]{{{'expl_1/after_explain_1.png'}}}",
                options=TikZOptions({"right": f"of n11"}),
                handle="n12"))    
    
        for r in range(2, 6):
            pic.append(TikZNode(
                        text=rf"\includegraphics[width={0.8*m_width}px]{{expl_1/mol_{r}.png}}",
                        options=TikZOptions({"below": f"of n{r-1}1"}),
                        handle=f"n{r}1"))
            pic.append(TikZNode(
                    text=rf"\includegraphics[width={m_width}px]{{expl_1/after_explain_{r}.png}}",
                    options=TikZOptions({"below": f"of n{r-1}2"}),
                    handle=f"n{r}2"))    

    doc.generate_pdf(out_file, clean_tex=False)


os.makedirs("expl_1", exist_ok=True)

data_info = MOLECULAR_DATASETS['qm9']

loaders = load_dataset('qm9xas_canonical', 100, [0.8, 0.1, 0.1])
loader = loaders['loader_tst'] 
transform = loaders['transform']
batch = next(iter(loader))
id = 1
x, a, spec, smile = batch['x'][id], batch['a'][id], batch['spec'][id], batch['s'][id]
e_min = 276
e_max = 280


energies = torch.linspace(MIN_E, MAX_E, N_GRID)

path_model = '/home/rektomar/projects/XAS-PGC/spectra_pred/results/trn/ckpt/qm9xas/ffnn_zero_sort_mask/dataset=qm9xas_order=canonical_model=ffnn_zero_sort_mask_nd_n=9_nk_n=5_nk_e=4_nd_y=100_nl=8_device=cuda_lr=0.01_betas=[0.9, 0.999]_weight_decay=0.0_transform=normal_num_epochs=100_batch_size=256_seed=0.pt'
model = torch.load(path_model, weights_only=False)

pred_spec = model.predict(x.unsqueeze(0).to(model.device), a.unsqueeze(0).to(model.device)).squeeze(0).detach().cpu()


### Target and prediction (entire molecule)
target = transform.inverse(spec)
prediction = transform.inverse(pred_spec)

plt.figure()
plt.plot(energies, target, label='GT Spectrum', color='blue')
plt.plot(energies, prediction, label='Predicted Spectrum', color='orange')
plt.title('Prediction from the entire molecule')
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity')
plt.fill_between([e_min, e_max], [0, 0], [torch.max(target), torch.max(target)], alpha=0.4)

plt.legend()
plt.savefig('expl_1/before_explain.png')
plt.close()



e_mask = choose_range(e_min, e_max)

# Explain

for i in range(1, 6):

    results = bruteforce_search(model.cpu(), x, a, e_mask)
    m = torch.tensor(results[i][0]) # get mask

    mol, hit_atoms, hit_bonds = g2mol_e(x, a, data_info['atom_list'], m)


    img = Draw.MolToImage(
        mol,
        highlightAtoms=hit_atoms,
        highlightBonds=hit_bonds,
    )

    img.save(f'expl_1/mol_{i}.png')


    ### Target and prediction (entire molecule)

    x_m, a_m = mask_graph(x, a, m)

    pred_spec_expl = model.predict(x_m.unsqueeze(0).to(model.device), a_m.unsqueeze(0).to(model.device)).squeeze(0).detach().cpu()


    target = transform.inverse(spec)
    prediction_expl = transform.inverse(pred_spec_expl)

    plt.figure()
    plt.plot(energies, target, label='GT Spectrum', color='blue')
    plt.plot(energies, prediction_expl, label='Predicted Spectrum', color='orange')
    plt.title('Prediction from submolecule')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.fill_between([e_min, e_max], [0, 0], [torch.max(target), torch.max(target)], alpha=0.4)

    plt.legend()
    plt.savefig(f'expl_1/after_explain_{i}.png')
    plt.close()


make_pdf_from_grid(f"expl_grid_{id}")

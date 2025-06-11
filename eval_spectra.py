import torch
import matplotlib.pyplot as plt

from utils.spec_datasets import load_dataset, MOLECULAR_DATASETS
from utils.molecular import gs2mols, mols2smls, get_vmols, g2mol

from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage, MolToImage


energies = torch.linspace(270, 300, 100)
atom_list = MOLECULAR_DATASETS['qm9']['atom_list']

def graph2spectrum(model, x, a, gt_spec):
    m, s = model.predict_spectrum(x, a)

    mu = m.cpu().detach().squeeze()
    sigma = s.cpu().detach().squeeze()

    plt.figure()
    plt.plot(energies, mu, label='Predicted Spectrum', color='blue')
    # plt.fill_between(energies, mu - sigma, mu + sigma, color='blue', alpha=0.3)
    plt.plot(energies, gt_spec.cpu().detach().squeeze(), label='GT Spectrum', color='orange')
    plt.xlabel('Energy')
    plt.legend()

    plt.savefig('graph2spectrum.png')
    plt.close()


def plot_and_save_spectrum(energies, spectrum, filename='spectrum.png'):
    plt.figure()
    plt.plot(energies, spectrum, color='blue')
    plt.xlabel('Energy')
    plt.savefig(filename)
    plt.close()


def spectrum2graph(model, x, a, spec, num_samples=50):
    x_sam, a_sam = model.sample_given_spectrum(spec.expand(num_samples, -1))

    mols = gs2mols(x_sam, a_sam, atom_list)
    smls = mols2smls(mols)
    vmols, vsmls = get_vmols(smls)

    gt_mol = g2mol(x.squeeze(0), a.squeeze(0), atom_list)
    gt_sml = Chem.MolToSmiles(gt_mol, canonical=True)

    plot_and_save_spectrum(energies, spec.squeeze(0).cpu().detach(), filename='input_spectrum.png')

    img_sam = MolsToGridImage(mols=vmols, molsPerRow=5, subImgSize=(400, 400), useSVG=False)
    img_sam.save('molecules_given_spectrum.png')

    img_gt = MolToImage(gt_mol, size=(400, 400))
    img_gt.save('gt_molecule.png')

def main():
    loaders = load_dataset('qm9xas', 256, [0.8, 0.1, 0.1])

    model_path = '/home/rektomar/projects/XAS-PGC/results/trn/ckpt/qm9xas/pgc_ffnn_spec/dataset=qm9xas_order=canonical_model=pgc_ffnn_spec_nd_n=9_nd_s=100_nk_n=5_nk_e=4_nz=32_nh=2048_nl_b=5_nl_n=3_nl_e=3_nl_s=3_nb=16384_nc=2_device=cuda_lr=0.001_betas=[0.9, 0.999]_weight_decay=0.0_num_epochs=100_batch_size=256_seed=0.pt'
    model = torch.load(model_path, weights_only=False)
    model.eval()

    batch = next(iter(loaders['loader_trn']))

    id = 100
    x = batch['x'][id:id+1].to('cuda')
    a = batch['a'][id:id+1].to('cuda')
    spec = batch['spec'][id:id+1].to('cuda')
    graph2spectrum(model, x, a, spec)
    spectrum2graph(model, x, a, spec)

if __name__ == '__main__':
    main()
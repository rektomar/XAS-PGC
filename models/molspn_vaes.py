import torch
import torch.nn as nn

from typing import Optional
from models.utils import CategoricalDecoder, GaussianEncoder, EncoderFFNN , DecoderFFNN, GaussianSampler, GraphXCoder
from models.graph_transformer import GraphTransformer


# def log_prob(self, x: torch.Tensor, n_mc_samples: int = 1, n_chunks: int = None):
#     # Compute KL divergence
#     mu, logvar = self.encoder(x).split(self.nh, dim=1)
#     kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=[i+1 for i in range(x.ndim-1)])

#     # Compute reconstruction error with n_mc_samples
#     mu = mu.repeat_interleave(n_mc_samples, 0)
#     logvar = logvar.repeat_interleave(n_mc_samples, 0)
#     std = torch.exp(0.5 * logvar)
#     z = mu + std * torch.randn_like(std)

#     x_recon = []
#     z_chunks = tuple([z]) if n_chunks is None else z.split(int(z.size(0) / n_chunks), 0)
#     for z_chunk in z_chunks:
#         x_recon.append(self.decoder(z_chunk))
#     x_recon = torch.cat(x_recon, dim=0)

#     all_recon = self.recon_loss(x_recon, x.repeat_interleave(n_mc_samples, 0))
#     recon = all_recon.view(x.shape[0], n_mc_samples, -1).sum(dim=[i+2 for i in range(x.ndim-1)])

#     # Compute log_prob and average over n_mc_samples
#     log_prob = -(kld[..., None] + recon)
#     log_prob = log_prob.logsumexp(dim=1) - np.log(n_mc_samples)

#     return log_prob


class MolSPNVAEFSort(nn.Module):
    def __init__(self,
                 nd_nx: int,
                 nk_nx: int,
                 nk_ex: int,
                 nd_nz: int,
                 nk_nz: int,
                 nk_ez: int,
                 nh_n: int,
                 nh_e: int,
                 nl_b: int,
                 nl_n: int,
                 nl_e: int,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNVAEFSort, self).__init__()

        encoder_network = EncoderFFNN(nd_nx, nk_nx, nk_ex, nd_nz, 2*nk_nz, 2*nk_ez, nh_n, nh_e, nl_b, nl_n, nl_e, device=device)
        decoder_network = DecoderFFNN(nd_nz, nk_nz, nk_ez, nd_nx,   nk_nx,   nk_ex, nh_n, nh_e, nl_b, nl_n, nl_e, device=device)

        self.encoder = GaussianEncoder(   encoder_network,  device=device)
        self.decoder = CategoricalDecoder(decoder_network,  device=device)
        self.sampler = GaussianSampler(nd_nz, nk_nz, nk_ez, device=device)
        self.device = device
        self.to(device)

    def forward(self, x, a):
        x_node = x.to(device=self.device, dtype=torch.float)
        x_edge = a.to(device=self.device, dtype=torch.float)

        kld_loss, z_node, z_edge = self.encoder(x_node, x_edge)
        rec_loss = self.decoder(x_node, x_edge, z_node, z_edge)

        return rec_loss.sum(dim=1) - kld_loss.sum(dim=1)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        z_node, z_edge, _ = self.sampler(num_samples)
        x_node, x_edge = self.decoder.sample(z_node, z_edge)
        return x_node, x_edge

class MolSPNVAEXSort(nn.Module):
    def __init__(self,
                 nk_nx: int,
                 nk_ex: int,
                 nd_nz: int,
                 nk_nz: int,
                 nk_ez: int,
                 nh_n: int,
                 nh_e: int,
                 nl_n: int,
                 nl_e: int,
                 nl_m: int,
                 nb: int,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNVAEXSort, self).__init__()

        encoder_network = GraphXCoder(nk_nx, nk_ex, 2*nk_nz, 2*nk_ez, nh_n, nh_e, nl_n, nl_e, nl_m, nb, device=device)
        decoder_network = GraphXCoder(nk_nz, nk_ez,   nk_nx,   nk_ex, nh_n, nh_e, nl_n, nl_e, nl_m, nb, device=device)

        self.encoder = GaussianEncoder(   encoder_network,  device=device)
        self.decoder = CategoricalDecoder(decoder_network,  device=device)
        self.sampler = GaussianSampler(nd_nz, nk_nz, nk_ez, device=device)
        self.device = device
        self.to(device)

    def forward(self, x, a):
        x_node = x.to(device=self.device, dtype=torch.float)
        x_edge = a.to(device=self.device, dtype=torch.float)

        kld_loss, z_node, z_edge = self.encoder(x_node, x_edge)
        rec_loss = self.decoder(x_node, x_edge, z_node, z_edge)

        return rec_loss.sum(dim=1) - kld_loss.sum(dim=1)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        z_node, z_edge, _ = self.sampler(num_samples)
        x_node, x_edge = self.decoder.sample(z_node, z_edge)
        return x_node, x_edge

class MolSPNVAETSort(nn.Module):
    def __init__(self,
                 nk_nx: int,
                 nk_ex: int,
                 nd_nz: int,
                 nk_nz: int,
                 nk_ez: int,
                 nl: int,
                 nh_n: int,
                 nh_e: int,
                 mh_n: int,
                 mh_e: int,
                 n_head: int,
                 df_n: int,
                 df_e: int,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNVAETSort, self).__init__()

        encoder_network = GraphTransformer(nl, nk_nx, nk_ex, 2*nk_nz, 2*nk_ez, mh_n, mh_e, n_head, nh_n, nh_e, df_n, df_e, device=device)
        decoder_network = GraphTransformer(nl, nk_nx, nk_ex,   nk_nz,   nk_ez, mh_n, mh_e, n_head, nh_n, nh_e, df_n, df_e, device=device)

        self.encoder = GaussianEncoder(   encoder_network,  device=device)
        self.decoder = CategoricalDecoder(decoder_network,  device=device)
        self.sampler = GaussianSampler(nd_nz, nk_nz, nk_ez, device=device)
        self.device = device
        self.to(device)

    def forward(self, x, a):
        x_node = x.to(device=self.device, dtype=torch.float)
        x_edge = a.to(device=self.device, dtype=torch.float)

        kld_loss, z_node, z_edge = self.encoder(x_node, x_edge)
        rec_loss = self.decoder(x_node, x_edge, z_node, z_edge)

        return rec_loss.sum(dim=1) - kld_loss.sum(dim=1)

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        z_node, z_edge, _ = self.sampler(num_samples)
        x_node, x_edge = self.decoder.sample(z_node, z_edge)
        return x_node, x_edge

MODELS = {
    'molspn_vaef_sort': MolSPNVAEFSort,
    'molspn_vaex_sort': MolSPNVAEXSort,
    'molspn_vaet_sort': MolSPNVAETSort,
}

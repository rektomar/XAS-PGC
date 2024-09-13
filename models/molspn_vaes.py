import torch
import torch.nn as nn
from torch.distributions import Categorical

from typing import Optional
from models.utils import CategoricalDecoder, GaussianEncoder, EncoderFFNN , DecoderFFNN, GaussianSampler, GraphXCoder
from models.graph_transformer import GraphTransformer
from utils.graph_features_general import ExtraFeatures

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
                 nx: int,
                 nx_x: int,
                 nx_a: int,
                 nz: int,
                 nz_x: int,
                 nz_a: int,
                 nz_y: int,
                 h_x: int,
                 h_a: int,
                 h_y: int,
                 l_x: int,
                 l_a: int,
                 l_y: int,
                 l_b: int,
                 ftype: str,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNVAEFSort, self).__init__()

        self.features_g = ExtraFeatures(ftype, nx)

        nf_x = self.features_g.nf_x
        nf_a = self.features_g.nf_a
        nf_y = self.features_g.nf_y

        encoder_network = EncoderFFNN(
            nx, nz, nx_x+nf_x, nx_a+nf_a, nf_y, 2*nz_x, 2*nz_a, 2*nz_y, h_x, h_a, h_y, l_x, l_a, l_y, l_b, device=device
        )
        decoder_network = DecoderFFNN(
            nz, nx, nz_x, nz_a, nz_y, nx_x, nx_a, nf_y, h_x, h_a, h_y, l_x, l_a, l_y, l_b, device=device
        )

        self.encoder = GaussianEncoder(   encoder_network,   device=device)
        self.decoder = CategoricalDecoder(decoder_network,   device=device)
        self.sampler = GaussianSampler(nz, nz_x, nz_a, nz_y, device=device)
        self.device = device
        self.to(device)

    def forward(self, x, a):
        xx = x.to(device=self.device, dtype=torch.float)
        aa = a.to(device=self.device, dtype=torch.float)

        xf, af, yf = self.features_g(xx, aa)
        xe = torch.cat((xx, xf), dim=-1)
        ae = torch.cat((aa, af), dim=-1)

        kld_loss, zx, za, zy = self.encoder(xe, ae, yf)
        rec_loss = self.decoder(xx, aa, [], zx, za, zy)

        return rec_loss - 1e-6*kld_loss

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        zx, za, zy, _ = self.sampler(num_samples)
        xx, xa = self.decoder.sample(zx, za, zy)
        return xx, xa

    def sample_conditional(self, x_node, x_edge, m_node, m_edge, n_mc_samples=16384):
        # interpreting decoder as continuous mixture
        # TODO: move this function to Decoder
        # TODO: add mum_samples
        z_node, z_edge, w = self.sampler(n_mc_samples)

        # marginal probs for each component
        ll_marg = self.decoder.cm_logpdf_marginal(x_node, x_edge, m_node, m_edge, z_node, z_edge) 
        k = Categorical(logits=w+ll_marg).sample() # add num_samples here 
        xc_node, xc_edge = self.decoder.sample(z_node[k], z_edge[k])

        xc_node[m_node] = x_node[m_node].cpu()
        xc_edge[m_edge] = x_edge[m_edge].cpu()

        return xc_node, xc_edge

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
                 nx: int,
                 nx_x: int,
                 nx_a: int,
                 nz: int,
                 nz_x: int,
                 nz_a: int,
                 nz_y: int,
                 nl: int,
                 nh_x: int,
                 nh_a: int,
                 nh_y: int,
                 mh_x: int,
                 mh_a: int,
                 mh_y: int,
                 n_head: int,
                 df_x: int,
                 df_a: int,
                 df_y: int,
                 ftype: str,
                 device: Optional[str]='cuda'
                 ):
        super(MolSPNVAETSort, self).__init__()

        self.features_g = ExtraFeatures(ftype, nx)

        nf_x = self.features_g.nf_x
        nf_a = self.features_g.nf_a
        nf_y = self.features_g.nf_y

        encoder_network = GraphTransformer(
            nl, nx_x+nf_x, nx_a+nf_a, nf_y, 2*(nz_x+nf_x), 2*(nz_a+nf_a), 2*nf_y, mh_x, mh_a, mh_y, n_head, nh_x, nh_a, nh_y, df_x, df_a, df_y, device=device
        )
        decoder_network = GraphTransformer(
            nl, nz_x+nf_x, nz_a+nf_a, nf_y, nx_x, nx_a, 0, mh_x, mh_a, mh_y, n_head, nh_x, nh_a, nh_y, df_x, df_a, df_y, device=device
        )

        self.encoder = GaussianEncoder(encoder_network, device=device)
        self.decoder = CategoricalDecoder(decoder_network, device=device)
        self.sampler = GaussianSampler(nz, nz_x+nf_x, nz_a+nf_a, nf_y, device=device)
        self.device = device
        self.to(device)

    def forward(self, x, a):
        xx = x.to(device=self.device, dtype=torch.float)
        aa = a.to(device=self.device, dtype=torch.float)

        xf, af, yf = self.features_g(xx, aa)
        xe = torch.cat((xx, xf), dim=-1)
        ae = torch.cat((aa, af), dim=-1)

        kld_loss, zx, za, zy = self.encoder(xe, ae, yf)
        rec_loss = self.decoder(xx, aa, [], zx, za, zy)

        return rec_loss - 9e-1*kld_loss

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        zx, za, zy, _ = self.sampler(num_samples)
        xx, xa = self.decoder.sample(zx, za, zy)
        return xx, xa

MODELS = {
    'molspn_vaef_sort': MolSPNVAEFSort,
    'molspn_vaex_sort': MolSPNVAEXSort,
    'molspn_vaet_sort': MolSPNVAETSort,
}

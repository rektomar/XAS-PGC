import torch
import torch.nn as nn

from typing import Optional
from models.utils import CategoricalEncoder, CategoricalDecoder, GaussianEncoder, EncoderFFNNTril, DecoderFFNNTril, EncoderFFNN, DecoderFFNN, GaussianSampler, CategoricalSampler
from models.graph_transformer import GraphTransformer
from utils.graph_features_general import ExtraFeatures


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

        # encoder_network = EncoderFFNN(
        #     nx, nz, nx_x+nf_x, nx_a+nf_a, nf_y, 2*nz_x, 2*nz_a, 2*nz_y, h_x, h_a, h_y, l_x, l_a, l_y, l_b, device=device
        # )
        # decoder_network = DecoderFFNN(
        #     nz, nx, nz_x, nz_a, nz_y, nx_x, nx_a, nf_y, h_x, h_a, h_y, l_x, l_a, l_y, l_b, device=device
        # )
        # self.encoder = GaussianEncoder(   encoder_network,   device=device)
        # self.decoder = CategoricalDecoder(decoder_network,   device=device)
        # self.sampler = GaussianSampler(nz, nz_x, nz_a, nz_y, device=device)

        encoder_network = EncoderFFNNTril(
            nx, nz, nx_x+nf_x, nx_a+nf_a, nf_y, nz_x, nz_a, 2*nz_y, h_x, h_a, h_y, l_x, l_a, l_y, l_b, device=device
        )
        decoder_network = DecoderFFNNTril(
            nz, nx, nz_x, nz_a, nz_y, nx_x, nx_a, nf_y, h_x, h_a, h_y, l_x, l_a, l_y, l_b, device=device
        )
        self.encoder = CategoricalEncoder(encoder_network,      device=device)
        self.decoder = CategoricalDecoder(decoder_network,      device=device)
        self.sampler = CategoricalSampler(nz, nz_x, nz_a, nz_y, device=device)

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

        return rec_loss - kld_loss

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        zx, za, zy, _ = self.sampler(num_samples)
        xx, xa = self.decoder.sample(zx, za, zy)
        return xx, xa

    def sample_conditional(self, x, a, mx, ma, num_samples, n_mc_samples=16384):
        # interpreting decoder as continuous mixture
        zx, za, zy, logw = self.sampler(n_mc_samples)
        
        # NOTE: it would make more sense to propagate 'num_samples'
        #       to Decoders sample function in the future
        x_r = x.repeat_interleave(num_samples, dim=0)
        a_r = a.repeat_interleave(num_samples, dim=0)
        mx_r = mx.repeat_interleave(num_samples, dim=0)
        ma_r = ma.repeat_interleave(num_samples, dim=0)

        xxc, xac = self.decoder.sample_conditional(x_r, a_r, mx_r, ma_r, zx, za, zy, logw)
        xxc = xxc.reshape(num_samples, *x.shape)
        xac = xac.reshape(num_samples, *a.shape)
        return xxc, xac

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
        # self.sampler = GaussianSampler(nz, nz_x+nf_x, nz_a+nf_a, nf_y, device=device)
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

        return rec_loss - 7e-1*kld_loss

    def logpdf(self, x, a):
        return self(x, a).mean()

    def sample(self, num_samples):
        zx, za, zy, _ = self.sampler(num_samples)
        xx, xa = self.decoder.sample(zx, za, zy)
        return xx, xa

MODELS = {
    'molspn_vaef_sort': MolSPNVAEFSort,
    'molspn_vaet_sort': MolSPNVAETSort,
}

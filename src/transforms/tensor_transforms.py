import torch
from torch import nn


class DimSwap(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(*self.permutation)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


def pad_psf(psf):
    h1 = psf.shape[-2] // 2
    w1 = psf.shape[-1] // 2
    pad_psf = torch.zeros(
        (psf.shape[0], psf.shape[1], psf.shape[2] * 2, psf.shape[3] * 2),
        dtype=psf.dtype,
        device=psf.device,
    )
    pad_psf[:, :, h1 : h1 + psf.shape[-2], w1 : w1 + psf.shape[-1]] = psf
    return pad_psf, h1, w1


def psf_prepare(psf):
    psf, h1, w1 = pad_psf(psf)
    psf = psf.clamp_min(0)
    psf = 5 * psf / psf.sum(dim=(-2, -1), keepdim=True)
    psf = torch.fft.ifftshift(psf, dim=(-2, -1))
    psf_fft = torch.fft.fft2(psf)
    return psf_fft, h1, w1

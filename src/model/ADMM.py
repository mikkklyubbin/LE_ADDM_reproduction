from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

from src.model.ADMM_math import (
    calc_norm,
    calc_norms,
    check_H,
    ct,
    make_iteration,
    zero_init,
)
from src.model.drunet import Drunet
from src.transforms.tensor_transforms import psf_prepare
from src.utils.io_utils import ROOT_PATH


def inverse_softplus(x):
    x = torch.as_tensor(x)
    return torch.log(torch.expm1(x))


def save_img(img, name):
    plt.imshow(img)
    Path("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug").mkdir(
        parents=True, exist_ok=True
    )
    plt.savefig(
        str(ROOT_PATH) + f"/data/debug/{name}", bbox_inches="tight", pad_inches=0
    )
    plt.close()


class ADMM(nn.Module):
    def __init__(self, num_its=100, tau=2 * 1e-4, us=1e-4) -> None:
        super().__init__()
        self.register_buffer("_device_indicator", torch.empty(0), persistent=False)
        self.tau = tau
        self.us = [us] * 4
        self.num_its = num_its

    @property
    def device(self):
        return self._device_indicator.device

    def forward(self, lensless, **batch):
        dtype = lensless.dtype
        device = self.device
        b = lensless.to(device)
        if "psf_fft" not in batch:
            psf = batch["psf"].to(device)
            psf_fft, h1, w1 = psf_prepare(psf)
        else:
            psf_fft = batch["psf_fft"].to(device)
            h1 = psf_fft.shape[-2] // 4
            w1 = psf_fft.shape[-1] // 4
        x_0, al1, al2_x, al2_y, al3 = zero_init(psf_fft, dtype, device)
        norm_psf, norm_dx, norm_dy = calc_norms(x_0, psf_fft)
        ctb = ct(b)
        mask = ct(torch.ones_like(b))
        for i in range(self.num_its):
            x_0, al1, al2_x, al2_y, al3 = make_iteration(
                x_0,
                al1,
                al2_x,
                al2_y,
                al3,
                psf_fft,
                b,
                self.us,
                self.tau,
                norm_psf,
                norm_dx,
                norm_dy,
                ctb,
                mask,
            )
        x_0 = x_0[:, :, h1 : h1 + lensless.shape[-2], w1 : w1 + lensless.shape[-1]]
        return {"reconstructed": x_0}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class leADMM(nn.Module):
    def __init__(self, num_its=20) -> None:
        super().__init__()
        self.us = nn.Parameter(
            inverse_softplus(torch.zeros((num_its, 4), requires_grad=True) + 1e-4)
        )
        self.tau = nn.Parameter(
            inverse_softplus(torch.zeros(num_its, requires_grad=True) + 2e-4)
        )
        self.num_its = num_its
        self.inited = False
        self.dx_norm = None
        self.dy_norm = None

    @property
    def device(self):
        return self.us.device

    def forward(self, lensless, **batch):
        dtype = lensless.dtype
        device = self.device
        b = lensless.to(device)
        if "psf_fft" not in batch:
            psf = batch["psf"].to(device)
            psf_fft, h1, w1 = psf_prepare(psf)
        else:
            psf_fft = batch["psf_fft"].to(device)
            h1 = psf_fft.shape[-2] // 4
            w1 = psf_fft.shape[-1] // 4
        x_0, al1, al2_x, al2_y, al3 = zero_init(psf_fft, dtype, device)
        ctb = ct(b)
        mask = ct(torch.ones_like(b))
        norm_psf = 0
        if not (self.inited):
            norm_psf, self.dx_norm, self.dy_norm = calc_norms(x_0, psf_fft)
            self.inited = True
        else:
            norm_psf = calc_norm(psf_fft)
        for i in range(self.num_its):
            x_0, al1, al2_x, al2_y, al3 = make_iteration(
                x_0,
                al1,
                al2_x,
                al2_y,
                al3,
                psf_fft,
                b,
                F.softplus(self.us[i]),
                F.softplus(self.tau[i]),
                norm_psf,
                self.dx_norm,
                self.dy_norm,
                ctb,
                mask,
            )
        x_0 = x_0[:, :, h1 : h1 + lensless.shape[-2], w1 : w1 + lensless.shape[-1]]
        return {"reconstructed": x_0}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class ADMM_plus_DRU(nn.Module):
    def __init__(
        self, num_its=20, drunet_channels=[64, 128, 256], use_start=False, use_end=False
    ) -> None:
        super().__init__()
        self.admm = leADMM(num_its)
        self.pred = nn.Identity()
        self.post = nn.Identity()
        if use_start:
            self.pred = Drunet(in_channels=3, channels=drunet_channels)
        if use_end:
            self.post = Drunet(in_channels=3, channels=drunet_channels)

    def forward(self, lensless, **batch):
        lensless = self.pred(lensless)
        addm_out = self.admm(lensless, **batch)
        addm_out["reconstructed"] = self.post(addm_out["reconstructed"])
        return addm_out

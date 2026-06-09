import torch
from torch import nn

from src.model.ADMM_math import calc_norms, make_iteration, zero_init, check_H
from src.model.drunet import Drunet
from matplotlib import pyplot as plt
from pathlib import Path
from src.utils.io_utils import ROOT_PATH
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

def save_img(img, name):
    plt.imshow(img)
    Path("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug").mkdir(parents=True, exist_ok=True)
    plt.savefig( str(ROOT_PATH) + f"/data/debug/{name}", bbox_inches="tight", pad_inches=0)
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

    def forward(self, lensless, psf, **batch):
        print(lensless.shape)
        print(psf.shape)
        dtype = lensless.dtype
        device = self.device
        b = lensless.to(device)
        psf = psf.to(device)
        psf, h1, w1 = pad_psf(psf)
        psf = psf.clamp_min(0)
        # psf = 4 * psf / psf.sum(dim=(-2, -1), keepdim=True)
        psf = torch.fft.ifftshift(psf, dim=(-2, -1))
        psf_fft = torch.fft.fft2(psf)
        if ("lensed" in batch) :
            print(batch["lensed"].abs().max())
            print(b.abs().max())
            print(lensless.abs().max())
            rec_less = check_H(batch["lensed"].to(device), psf_fft)
            save_img(rec_less[0].permute((1, 2, 0)).cpu().numpy(), "check_H")
            rec_less = rec_less[:, :, h1 : h1 + lensless.shape[-2], w1 : w1 + lensless.shape[-1]]
            save_img(b[0].permute((1, 2, 0)).cpu().numpy(), "lensless")
            print((rec_less - b).abs().mean())
        x_0, al1, al2_x, al2_y, al3 = zero_init(psf_fft, dtype, device)
        norm_psf, norm_dx, norm_dy = calc_norms(x_0, psf_fft)
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
            )
        print(x_0.shape)
        print(x_0.max())
        plt.imshow(x_0[0].permute((1, 2, 0)).cpu().numpy())
        Path("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug").mkdir(parents=True, exist_ok=True)
        plt.savefig("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug/debug3", bbox_inches="tight", pad_inches=0)
        plt.close()
        x_0 = x_0[:, :, h1 : h1 + lensless.shape[-2], w1 : w1 + lensless.shape[-1]]
        img = x_0[0].detach().cpu()
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        plt.imshow(img.permute((1, 2, 0)).numpy())
        Path("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug").mkdir(parents=True, exist_ok=True)
        plt.savefig("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug/debug4", bbox_inches="tight", pad_inches=0)
        plt.close()
        x_0 = torch.clamp(x_0, 0, 1)
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
        self.us = nn.Parameter(torch.zeros((num_its, 4), requires_grad=True)  + 1e-4)
        self.tau = nn.Parameter(torch.zeros(num_its, requires_grad=True) + 2e-4)
        self.num_its = num_its

    @property
    def device(self):
        return self.us.device

    def forward(self, lensless, psf, **batch):
        dtype = lensless.dtype
        device = self.device
        b = lensless
        psf = torch.fft.fftshift(psf, dim=(-2, -1))
        psf_fft = torch.fft.fft2(psf)
        x_0, al1, al2_x, al2_y, al3 = zero_init(psf_fft, dtype, device)
        norm_psf, norm_dx, norm_dy = calc_norms(x_0, psf_fft)
        for i in range(self.num_its):
            x_0, al1, al2_x, al2_y, al3 = make_iteration(
                x_0,
                al1,
                al2_x,
                al2_y,
                al3,
                psf_fft,
                b,
                self.us[i],
                self.tau[i],
                norm_psf,
                norm_dx,
                norm_dy,
            )
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
    def __init__(self, num_its=20, drunet_channels = [64, 128, 256], use_start = False, use_end = False) -> None:
        super().__init__()
        self.admm = leADMM(num_its)
        self.pred = nn.Identity()
        self.post = nn.Identity()
        if use_start:
            self.pred = Drunet(in_channels=3, channels=drunet_channels)
        if use_end:
            self.post = Drunet(in_channels=3, channels=drunet_channels)

    def forward(self, lensless, psf, **batch):
        lensless = self.pred(lensless)
        addm_out = self.admm(lensless, psf, **batch)
        addm_out["reconstructed"] = self.post(addm_out["reconstructed"])
        return addm_out
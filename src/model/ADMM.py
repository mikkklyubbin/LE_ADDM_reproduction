import torch
from torch import nn
from src.model.ADMM_math import make_iteration, calc_norms, zero_init




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
        dtype = lensless.dtype
        device = self.device
        b = lensless.to(device)
        psf = psf.to(device)
        psf_fft = torch.fft.fft2(psf)
        x_0, al1, al2_x, al2_y, al3 = zero_init(psf_fft, dtype, device)
        norm_psf, norm_dx, norm_dy = calc_norms(x_0, psf_fft)
        for i in range(self.num_its):
            x_0, al1, al2_x, al2_y, al3 = make_iteration(
                x_0, al1, al2_x, al2_y, al3, psf_fft, b, self.us, self.tau, norm_psf, norm_dx, norm_dy
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




class leADMM(nn.Module):
    def __init__(self, num_its=20) -> None:
        super().__init__()
        self.us = torch.zeros((num_its, 4), requires_grad=True)
        self.tau = torch.zeros(num_its, requires_grad=True)
        self.num_its = num_its



    def forward(self, lensless, psf, **batch):
        dtype = lensless.dtype
        device = self.device
        b = lensless.to(device)
        psf = psf.to(device)
        psf_fft = torch.fft.fft2(psf)
        x_0, al1, al2_x, al2_y, al3 = zero_init(psf_fft, dtype, device)
        norm_psf, norm_dx, norm_dy = calc_norms(x_0, psf_fft)
        for i in range(self.num_its):
            x_0, al1, al2_x, al2_y, al3 = make_iteration(
                x_0, al1, al2_x, al2_y, al3, psf_fft, b, self.us[i], self.tau[i], norm_psf, norm_dx, norm_dy
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
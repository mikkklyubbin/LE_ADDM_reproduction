import torch
from torch import nn


def soft_threshold(z, threshold):
    return torch.sign(z) * torch.clamp(torch.abs(z) - threshold, min=0.0)


def H(psf_fft, x):
    return torch.fft.ifft2(psf_fft * torch.fft.fft2(x)).real


def H_T(psf_fft, y):
    return torch.fft.ifft2(torch.conj(psf_fft) * torch.fft.fft2(y)).real


def Dx(x):
    return torch.roll(x, shifts=-1, dims=-1) - x


def Dy(x):
    return torch.roll(x, shifts=-1, dims=-2) - x


def Dtx(x):
    return torch.roll(x, shifts=1, dims=-1) - x


def Dty(x):
    return torch.roll(x, shifts=1, dims=-2) - x


def calc_norm(x):
    return torch.abs(x) ** 2


class ADMM(nn.Module):
    def __init__(self, num_its=100, tau=2 * 1e-4, us=1e-4) -> None:
        super().__init__()
        self.tau = tau
        self.us = us
        self.num_its = num_its

    def make_iteration(self, x, al1, al2_x, al2_y, al3, psf_fft, b, reverse_op):
        u_x = soft_threshold(Dx(x) + al2_x / self.us, self.tau / self.us)
        u_y = soft_threshold(Dy(x) + al2_y / self.us, self.tau / self.us)
        v = (al1 + H(psf_fft, x) * self.us + b) / (1 + self.us)
        w = torch.clamp(al3 / self.us + x, 0)
        r = (
            (self.us * w - al3)
            + Dtx(self.us * u_x - al2_x)
            + Dty(self.us * u_y - al2_y)
            + H_T(psf_fft, self.us * v - al1)
        )
        x_n = torch.fft.ifft2(torch.fft.fft2(r) / reverse_op).real
        al1 += self.us * (H(psf_fft, x_n) - v)
        al2_x += self.us * (Dx(x_n) - u_x)
        al2_y += self.us * (Dy(x_n) - u_y)
        al3 += self.us * (x_n - w)
        return x_n, al1, al2_x, al2_y, al3

    def forward(self, measurement, psf, **batch):
        dtype = measurement.dtype
        device = self.device
        b = measurement.to(device)
        psf = psf.to(device)
        psf_fft = torch.fft.fft2(psf)
        x_0 = torch.zeros(psf_fft.shape, dtype=dtype, device=device)
        al1 = torch.zeros_like(x_0)
        al2_x = torch.zeros_like(x_0)
        al2_y = torch.zeros_like(x_0)
        al3 = torch.zeros_like(x_0)
        Dx_op = torch.zeros(x_0.shape[-2:], dtype=dtype, device=device)
        Dy_op = torch.zeros(x_0.shape[-2:], dtype=dtype, device=device)
        Dx_op[0, 0] = -1
        Dx_op[0, -1] = 1
        Dy_op[0, 0] = -1
        Dy_op[-1, 0] = 1
        reverse_op = self.us * (
            calc_norm(psf_fft)
            + calc_norm(torch.fft.fft2(Dx_op))
            + calc_norm(torch.fft.fft2(Dy_op))
            + 1
        )
        for i in range(self.num_its):
            x_0, al1, al2_x, al2_y, al3 = self.make_iteration(
                x_0, al1, al2_x, al2_y, al3, psf_fft, b, reverse_op
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

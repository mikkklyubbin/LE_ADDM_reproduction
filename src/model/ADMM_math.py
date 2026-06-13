import torch


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


def calc_norms(x_0, psf_fft):
    Dx_op = torch.zeros(x_0.shape[-2:], dtype=x_0.dtype, device=x_0.device)
    Dy_op = torch.zeros(x_0.shape[-2:], dtype=x_0.dtype, device=x_0.device)
    Dx_op[0, 0] = -1
    Dx_op[0, -1] = 1
    Dy_op[0, 0] = -1
    Dy_op[-1, 0] = 1
    return (
        calc_norm(psf_fft),
        calc_norm(torch.fft.fft2(Dx_op)),
        calc_norm(torch.fft.fft2(Dy_op)),
    )


def ct(b):
    h1 = b.shape[-2] // 2
    w1 = b.shape[-1] // 2
    pad_b = torch.zeros(
        (b.shape[0], b.shape[1], b.shape[2] * 2, b.shape[3] * 2),
        dtype=b.dtype,
        device=b.device,
    )
    pad_b[:, :, h1 : h1 + b.shape[-2], w1 : w1 + b.shape[-1]] = b
    return pad_b


def make_iteration(
    x,
    al1,
    al2_x,
    al2_y,
    al3,
    psf_fft,
    b,
    us,
    tau,
    norm_psf,
    norm_dx,
    norm_dy,
    ctb,
    mask,
):
    reverse_op = norm_psf * us[0] + norm_dx * us[1] + norm_dy * us[2] + us[3]
    u_x = soft_threshold(Dx(x) + al2_x / us[1], tau)
    u_y = soft_threshold(Dy(x) + al2_y / us[2], tau)
    v = (al1 + H(psf_fft, x) * us[0] + ctb) / (mask + us[0])
    w = torch.clamp(al3 / us[3] + x, 0)
    r = (
        (us[3] * w - al3)
        + Dtx(us[1] * u_x - al2_x)
        + Dty(us[2] * u_y - al2_y)
        + H_T(psf_fft, us[0] * v - al1)
    )
    x_n = torch.fft.ifft2(torch.fft.fft2(r) / reverse_op).real
    al1 = al1 + us[0] * (H(psf_fft, x_n) - v)
    al2_x = al2_x + us[1] * (Dx(x_n) - u_x)
    al2_y = al2_y + us[2] * (Dy(x_n) - u_y)
    al3 = al3 + us[3] * (x_n - w)
    return x_n, al1, al2_x, al2_y, al3


def zero_init(psf_fft, dtype, device):
    x_0 = torch.zeros(psf_fft.shape, dtype=dtype, device=device)
    al1 = torch.zeros_like(x_0)
    al2_x = torch.zeros_like(x_0)
    al2_y = torch.zeros_like(x_0)
    al3 = torch.zeros_like(x_0)
    return x_0, al1, al2_x, al2_y, al3


def check_H(lensed, psf_fft):
    lensed = ct(lensed)
    out = H(psf_fft, lensed)
    return out

import typing

import torch

from srwarp import types

@torch.no_grad()
def get_omega(
        du: torch.Tensor,
        dv: torch.Tensor,
        uvx: torch.Tensor,
        uvy: torch.Tensor) -> torch.Tensor:

    num = 2 * (du[0] * du[1] + dv[0] * dv[1])
    den = uvx - uvy
    omega = 0.5 * torch.atan2(num, den)
    return omega

@torch.no_grad()
def get_ab(
        omega: torch.Tensor,
        du: torch.Tensor,
        dv: torch.Tensor,
        uvx: torch.Tensor,
        uvy: torch.Tensor,
        regularize: bool=True) -> types._TT:

    det = du[0] * dv[1] - dv[0] * du[1]
    det.pow_(2)

    cos = torch.cos(omega)
    cos.pow_(2)

    # cos(2x) = 2cos^2(x) - 1
    num = det * (2 * cos - 1)

    den_shared = cos * (uvx + uvy)
    den_a = den_shared - uvx
    den_b = den_shared - uvy

    a_inv = den_a / num
    b_inv = den_b / num

    a_inv.sqrt_()
    b_inv.sqrt_()

    if regularize:
        a_inv.clamp_(max=1)
        b_inv.clamp_(max=1)

    return a_inv, b_inv

@torch.no_grad()
def get_modulator(
        du: torch.Tensor,
        dv: torch.Tensor,
        regularize: bool=True) -> types._TTTT:

    uvx = du[0].pow(2) + dv[0].pow(2)
    uvy = du[1].pow(2) + dv[1].pow(2)

    omega = get_omega(du, dv, uvx, uvy)
    a_inv, b_inv = get_ab(omega, du, dv, uvx, uvy, regularize=regularize)

    omega = omega.view(-1, 1, 1)
    a_inv = a_inv.view(-1, 1, 1)
    b_inv = b_inv.view(-1, 1, 1)

    cos = omega.cos()
    sin = omega.sin()

    mxx = a_inv * cos
    mxy = a_inv * sin
    myx = -b_inv * sin
    myy = b_inv * cos
    return mxx, mxy, myx, myy

@torch.no_grad()
def modulation(
        ox: torch.Tensor,
        oy: torch.Tensor,
        j: types._TT,
        regularize: bool=True) -> types._TT:

    mxx, mxy, myx, myy = get_modulator(*j, regularize=regularize)
    oxp = mxx * ox + mxy * oy
    oyp = myx * ox + myy * oy
    return oxp, oyp
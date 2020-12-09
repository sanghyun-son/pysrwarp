import math
import typing

import torch

from srwarp import transform
from srwarp import wtypes
from srwarp import debug

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
        regularize: bool=True) -> wtypes._TT:

    det = transform.determinant((du, dv))
    det.pow_(2)

    cos = torch.cos(omega)
    cos.pow_(2)

    # cos(2x) = 2cos^2(x) - 1
    den = det * (2 * cos - 1)

    num_shared = cos * (uvx + uvy)
    a = num_shared - uvx
    b = num_shared - uvy
    a /= den
    b /= den
    a.abs_()
    b.abs_()
    a.sqrt_()
    b.sqrt_()
    if regularize:
        a.clamp_(max=1)
        b.clamp_(max=1)

    return a, b

@torch.no_grad()
def align(omega: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> wtypes._TTT:
    quad = math.pi / 4
    range_1 = torch.logical_and(omega >= -quad, omega < quad).float()
    range_2 = torch.logical_and(omega >= quad, omega < 3 * quad).float()
    range_31 = (omega >= 3 * quad).float()
    range_32 = (omega < -3 * quad).float()
    range_4 = torch.logical_and(omega >= -3 * quad, omega < -quad).float()

    omega_new = range_1 * omega
    omega_new += range_2 * (omega - math.pi / 2)
    omega_new += range_31 * (omega - math.pi)
    omega_new += range_32 * (omega + math.pi)
    omega_new += range_4 * (omega + math.pi / 2)

    a_new = (range_1 + range_31 + range_32) * a
    a_new += (range_2 + range_4) * b

    b_new = (range_1 + range_31 + range_32) * b
    b_new += (range_2 + range_4) * a

    return omega_new, a_new, b_new

@torch.no_grad()
def get_modulator(
        du: torch.Tensor,
        dv: torch.Tensor,
        regularize: bool=True,
        dump: typing.Optional[dict]=None) -> wtypes._TTTT:

    uvx = du[0].pow(2) + dv[0].pow(2)
    uvy = du[1].pow(2) + dv[1].pow(2)

    omega = get_omega(du, dv, uvx, uvy)
    a, b = get_ab(omega, du, dv, uvx, uvy, regularize=regularize)
    # Optional?
    #omega, a, b = align(omega, a, b)

    if dump is not None:
        debug.dump_variable(dump, 'omega', omega)
        debug.dump_variable(dump, 'a', a)
        debug.dump_variable(dump, 'b', b)

    omega = omega.view(-1, 1, 1)
    a = a.view(-1, 1, 1)
    b = b.view(-1, 1, 1)

    cos = omega.cos()
    sin = omega.sin()

    mxx = a * cos
    mxy = a * sin
    myx = -b * sin
    myy = b * cos
    return mxx, mxy, myx, myy

@torch.no_grad()
def get_adaptive_coordinate(
        ox: torch.Tensor,
        oy: torch.Tensor,
        du: torch.Tensor,
        dv: torch.Tensor,
        regularize: bool=True,
        dump: typing.Optional[dict]=None) -> wtypes._TT:

    mxx, mxy, myx, myy = get_modulator(du, dv, regularize=regularize, dump=dump)
    oxp = mxx * ox + mxy * oy
    oyp = myx * ox + myy * oy
    return oxp, oyp

@torch.no_grad()
def modulation(
        ox: torch.Tensor,
        oy: torch.Tensor,
        j: wtypes._TT,
        regularize: bool=True,
        eps: float=1e-12,
        dump: typing.Optional[dict]=None) -> wtypes._TT:

    oxp, oyp = get_adaptive_coordinate(
        ox, oy, *j, regularize=regularize, dump=dump,
    )
    # Optimized for better memory usage
    oxp.pow_(2)
    oyp.pow_(2)
    num = oxp + oyp
    den = ox.pow(2) + oy.pow(2)
    # Zero-handling
    origin = (den < eps).float()
    den *= (1 - origin)
    origin *= eps
    den += origin

    num /= den
    num.sqrt_()
    oxp = num * ox
    oyp = num * oy

    if dump is not None:
        debug.dump_variable(dump, 'oxp', oxp)
        debug.dump_variable(dump, 'oyp', oyp)

    return oxp, oyp
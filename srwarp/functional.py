import math
import typing

import numpy as np

from srwarp import wtypes

import torch

@torch.no_grad()
def sine(a: float=2, t: float=12) -> typing.Callable:
    def _sine(xp: torch.Tensor, yp: torch.Tensor) -> wtypes._TT:
        x = xp
        y = yp - a * torch.sin(x / t) - a
        return x, y

    return _sine

@torch.no_grad()
def barrel(hp: int=512, wp: int=512, k: float=1) -> typing.Callable:
    def _barrel(xp: torch.Tensor, yp: torch.Tensor) -> wtypes._TT:
        hpc = (hp - 1) / 2
        wpc = (wp - 1) / 2
        xd = (xp - wpc) / wp
        yd = (yp - hpc) / hp
        rp_pow = xd.pow(2) + yd.pow(2)
        rp = rp_pow.sqrt()
        r = rp * (1 + k * rp_pow)
        factor = r / (rp + 1e-6)
        x = factor * (xp - wpc) + (hp / 2 - 1)
        y = factor * (yp - hpc) + (wp / 2 - 1)
        return x, y

    return _barrel

@torch.no_grad()
def spiral(hp: int=512, wp: int=512, k: float=1) -> typing.Callable:
    def _spiral(xp: torch.Tensor, yp: torch.Tensor) -> wtypes._TT:
        s = math.sqrt(hp ** 2 + wp ** 2)
        sx = (s - wp) / 2
        sy = (s - hp) / 2

        sc = (s - 1) / 2
        xd = (xp - sc) / wp
        yd = (yp - sc) / hp
        rp = xd.pow(2) + yd.pow(2)
        cos = torch.cos(k * rp)
        sin = torch.sin(k * rp)
        xt = (xp - sc) * cos + (yp - sc) * sin + sc
        yt = -(xp - sc) * sin + (yp - sc) * cos + sc
        x = xt - sx
        y = yt - sy
        return x, y

    return _spiral

@torch.no_grad()
def calibration(
        m: typing.Union[torch.Tensor, np.array],
        k1: float,
        k2: float,
        p1: float,
        p2: float,
        k3: float=0,
        k4: float=0,
        k5: float=0,
        k6: float=0,
        s1: float=0,
        s2: float=0,
        s3: float=0,
        s4: float=0,
        offset_x: float=0,
        offset_y: float=0) -> typing.Callable:

    cx = m[0, 2]
    cy = m[1, 2]
    fx = m[0, 0]
    fy = m[1, 1]
    def _calibration(xp: torch.Tensor, yp: torch.Tensor) -> wtypes._TT:
        x = (xp - cx - offset_x) / fx
        y = (yp - cy - offset_y) / fy
        xy = x * y
        x2 = x.pow(2)
        y2 = y.pow(2)
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2
        coeffs_num = 1 + k1 * r2 + k2 * r4 + k3 * r6
        coeffs_den = 1 + k4 * r2 + k5 * r4 + k6 * r6
        coeffs = coeffs_num / coeffs_den
        xx = x * coeffs + 2 * p1 * xy + p2 * (r2 + 2 * x2) + s1 * r2 + s2 * r4
        yy = y * coeffs + p1 * (r2 + 2 * y2) + 2 * p2 * xy + s3 * r2 + s4 * r4
        map_x = xx * fx + cx
        map_y = yy * fy + cy
        return map_x, map_y

    return _calibration


@torch.no_grad()
def scaling(f: typing.Callable, scale: float=1) -> typing.Callable:
    def _scaling(xp: torch.Tensor, yp: torch.Tensor) -> wtypes._TT:
        x, y = f(xp, yp)
        scale_inv = 1 / scale
        x *= scale_inv
        y *= scale_inv

        x += 0.5 * (scale_inv - 1)
        y += 0.5 * (scale_inv - 1)
        return x, y

    return _scaling

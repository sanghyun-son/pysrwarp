import math
import random
import typing

import torch
from torch import cuda

from srwarp import wtypes
from srwarp import grid


@torch.no_grad()
def inverse_3x3(m: torch.Tensor) -> torch.Tensor:
    '''
    Hard-coded matrix inversion for 3x3 matrices.

    Args:
        m (torch.Tensor): (3, 3) transformation matrix.

    Return:
        torch.Tensor: m^{-1}, which is calculated from cofactors.
    '''
    n = m.cpu().numpy()
    cofactor_00 = n[1, 1] * n[2, 2] - n[1, 2] * n[2, 1]
    cofactor_01 = n[1, 2] * n[2, 0] - n[1, 0] * n[2, 2]
    cofactor_02 = n[1, 0] * n[2, 1] - n[1, 1] * n[2, 0]
    cofactor_10 = n[0, 2] * n[2, 1] - n[0, 1] * n[2, 2]
    cofactor_11 = n[0, 0] * n[2, 2] - n[0, 2] * n[2, 0]
    cofactor_12 = n[0, 1] * n[2, 0] - n[0, 0] * n[2, 1]
    cofactor_20 = n[0, 1] * n[1, 2] - n[0, 2] * n[1, 1]
    cofactor_21 = n[0, 2] * n[1, 0] - n[0, 0] * n[1, 2]
    cofactor_22 = n[0, 0] * n[1, 1] - n[0, 1] * n[1, 0]
    # determinant
    d = n[0, 0] * cofactor_00 + n[0, 1] * cofactor_01 + n[0, 2] * cofactor_02

    if abs(d) < 1e-12:
        raise ValueError('Inverse matrix does not exist!')

    inv = torch.DoubleTensor([
        [cofactor_00, cofactor_10, cofactor_20],
        [cofactor_01, cofactor_11, cofactor_21],
        [cofactor_02, cofactor_12, cofactor_22],
    ])
    #inv = inv.to(dtype=m.dtype, device=m.device)
    inv /= d
    return inv

@torch.no_grad()
def transform_corners(
        x: torch.Tensor,
        m: torch.Tensor,
        digits_allowed: int=4) -> torch.Tensor:

    '''
    Assume that m is a DoubleTensor.
    '''
    h = x.size(-2)
    w = x.size(-1)
    # For higher accuracy
    #m = m.double()
    c = m.new_tensor([
        [-0.5, -0.5, w - 0.5, w - 0.5],
        [-0.5, h - 0.5, -0.5, h - 0.5],
        [1, 1, 1, 1],
    ])
    c = torch.matmul(m, c)
    c = c / c[-1, :]
    #factor = 10**digits_allowed
    #c = torch.trunc(factor * c) / factor
    return c

@torch.no_grad()
def compensate_matrix(
        x: torch.Tensor,
        m: torch.Tensor,
        exact: bool=False) -> typing.Tuple[torch.Tensor, wtypes._II, wtypes._II]:

    # (3, 4)
    c = transform_corners(x, m)

    def get_dimension(dim: int) -> wtypes._II:
        s_min = c[dim].min()
        s_max = c[dim].max()
        s_len = (s_max - s_min).ceil()
        s_len = int(s_len.item())
        if exact:
            s_offset = s_min
        else:
            s_offset = (s_min + 0.5).floor()
            s_offset = -int(s_offset.item())

        return s_len, s_offset

    x_len, x_offset = get_dimension(0)
    y_len, y_offset = get_dimension(1)
    m = compensate_offset(m, x_offset, y_offset, offset_first=False)
    return m, (y_len, x_len), (y_offset, x_offset)

@torch.no_grad()
def generate_transform(
        x: torch.Tensor,
        smin: float=0.35,
        smax: float=0.5,
        hmax: float=0.25,
        rmax: float=15,
        pmin: float=-0.6,
        pmax: float=0.6) -> torch.Tensor:

    h = x.size(-2)
    w = x.size(-1)
    mh = random_sheering(hmax)
    mr = random_rotation(rmax)
    ms = random_scaling(smin, smax)
    mp = random_projection(pmin, pmax, -0.75, 0.125, h, w)

    m = torch.matmul(mh, mr)
    m = torch.matmul(m, ms)
    m = torch.matmul(m, mp)
    return m

@torch.no_grad()
def get_random_transform(
        x: torch.Tensor,
        size_limit: int=1024,
        max_iters: int=10,
        **kwargs) -> torch.Tensor:

    for _ in range(max_iters):
        m = generate_transform(x, **kwargs)
        m, sizes, _ = compensate_matrix(x, m)
        if sizes[0] < size_limit and sizes[1] < size_limit:
            return m

    msg = 'No valid transform! Use larger max_iters. (Current: {})'.format(
        max_iters,
    )
    print(msg)
    return m

@torch.no_grad()
def sheering(hx: float, hy: float) -> torch.Tensor:
    m = torch.DoubleTensor([
        [1, hx, 0],
        [hy, 1, 0],
        [0, 0, 1],
    ])
    return m

@torch.no_grad()
def random_sheering(hmax: float) -> torch.Tensor:
    hx = random.uniform(-hmax, hmax)
    hy = random.uniform(-hmax, hmax)
    m = sheering(hx, hy)
    return m

@torch.no_grad()
def compensate_offset(
        m: torch.Tensor,
        ix: float,
        iy: float,
        offset_first: bool=True) -> torch.Tensor:

    t = translation(ix, iy)
    if offset_first:
        m = torch.matmul(m, t)
    else:
        m = torch.matmul(t, m)

    return m

@torch.no_grad()
def translation(tx: float, ty: float) -> torch.Tensor:
    m = torch.DoubleTensor([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ])
    return m

@torch.no_grad()
def compensate_scale(
        m: torch.Tensor,
        sx: float,
        sy: typing.Optional[float]=None) -> torch.Tensor:

    s = scaling(sx, sy=sy)
    m = torch.matmul(m, s)
    return m

@torch.no_grad()
def scaling(sx: float, sy: typing.Optional[float]=None) -> torch.Tensor:
    if sy is None:
        sy = sx

    tx = 0.5 * (sx - 1)
    ty = 0.5 * (sy - 1)
    m = torch.DoubleTensor([
        [sx, 0, tx],
        [0, sy, ty],
        [0, 0, 1],
    ])
    return m

@torch.no_grad()
def random_scaling(smin: float, smax: float) -> torch.Tensor:
    sx = random.uniform(smin, smax)
    sy = random.uniform(smin, smax)
    m = scaling(sx, sy=sy)
    return m

@torch.no_grad()
def compensate_rotation(m: torch.Tensor, theta: float) -> torch.Tensor:
    r = rotation(theta)
    m = torch.matmul(m, r)
    return m

@torch.no_grad()
def rotation(theta: float) -> torch.Tensor:
    m = torch.DoubleTensor([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1],
    ])
    return m

def random_rotation(rmax: float) -> torch.Tensor:
    rmax = abs(rmax)
    rmax = min(rmax, 45)
    theta = random.gauss(0, rmax / 3)
    theta = max(min(theta, rmax), -rmax)
    theta = 2 * math.pi * (theta / 360)
    m = rotation(theta)
    return m

@torch.no_grad()
def projection(px: float, py: float, tx: float, ty: float):
    m = torch.DoubleTensor([
        [1, 0, tx],
        [0, 1, ty],
        [px, py, 1],
    ])
    return m

def random_projection(
        pmin: float,
        pmax: float,
        tmin: float,
        tmax: float,
        h: float,
        w: float) -> torch.Tensor:

    px = random.uniform(pmin / w, pmax / w)
    py = random.uniform(pmin / h, pmax / h)
    tx = random.uniform(w * tmin, w * tmax)
    ty = random.uniform(h * tmin, h * tmax)
    m = projection(px, py, tx, ty)
    return m

@torch.no_grad()
def jacobian(
        f: typing.Union[torch.Tensor, typing.Callable],
        sizes: typing.Tuple[int, int],
        yi: typing.Optional[torch.Tensor]=None,
        eps: float=0.5) -> wtypes._TT:

    '''
    J = [
        [du[0], dv[0]],
        [du[1], dv[1]]
    ]
    '''
    if isinstance(f, torch.Tensor):
        grid_function = grid.projective_grid

    # We do not need such high precisions for the Jacobian
    # Still, we use the double precision to ensure accuracy of the following
    # equations
    #f = f.float()
    dl = grid_function(f, sizes, eps_x=-eps)
    dr = grid_function(f, sizes, eps_x=eps)
    dt = grid_function(f, sizes, eps_y=-eps)
    db = grid_function(f, sizes, eps_y=eps)

    if yi is not None:
        dl = dl[..., yi]
        dr = dr[..., yi]
        db = db[..., yi]
        dt = dt[..., yi]

    # (2, N) each
    du = (dr - dl) / (2 * eps)
    dv = (db - dt) / (2 * eps)

    du = du.float()
    dv = dv.float()
    return du, dv

@torch.no_grad()
def determinant(j: wtypes._TT) -> torch.Tensor:
    du, dv = j
    det = du[0] * dv[1] - dv[0] * du[1]
    return det

@torch.no_grad()
def replicate_matrix(m: torch.Tensor, do_replicate: bool=True) -> torch.Tensor:
    if do_replicate:
        m = m.repeat(cuda.device_count(), 1)

    return m

import typing

import torch

from srwarp import types
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

    inv = torch.Tensor([
        [cofactor_00, cofactor_10, cofactor_20],
        [cofactor_01, cofactor_11, cofactor_21],
        [cofactor_02, cofactor_12, cofactor_22],
    ])
    inv = inv.to(dtype=m.dtype, device=m.device)
    inv /= d
    return inv

@torch.no_grad()
def transform_corners(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    h = x.size(-2)
    w = x.size(-1)
    # For higher accuracy
    m = m.double()
    c = m.new_tensor([
        [-0.5, -0.5, w - 0.5, w - 0.5],
        [-0.5, h - 0.5, -0.5, h - 0.5],
        [1, 1, 1, 1],
    ])
    c = torch.matmul(m, c)
    c = c / c[-1, :]
    return c

@torch.no_grad()
def scaling(sx: float, sy: typing.Optional[float]=None) -> torch.Tensor:
    if sy is None:
        sy = sx

    tx = 0.5 * (sx - 1)
    ty = 0.5 * (sy - 1)
    m = torch.Tensor([
        [sx, 0, tx],
        [0, sy, ty],
        [0, 0, 1],
    ])
    return m

@torch.no_grad()
def jacobian(
        f: typing.Union[torch.Tensor, typing.Callable],
        sizes: typing.Tuple[int, int],
        yi: typing.Optional[torch.Tensor]=None,
        eps: float=0.5) -> types._TT:

    '''
    J = [
        [du[0], dv[0]],
        [du[1], dv[1]]
    ]
    '''
    if isinstance(f, torch.Tensor):
        grid_function = grid.projective_grid

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
    dv = (dt - db) / (2 * eps)
    return du, dv
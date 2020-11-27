import typing

import torch

from srwarp import types
import srwarp_cuda

@torch.no_grad()
def projective_grid(
        m: torch.Tensor,
        sizes: types._II,
        eps_y: float=0,
        eps_x: float=0) -> torch.Tensor:

    '''
    Args:
        sizes (int, int): Target domain size.
        m (torch.Tensor): Target to source transform.
        eps_y (float, optional): Perturbation along y-axis.
        eps_x (float, optional): Perturbation alogn x-axis.
    '''
    # Must be done on GPU
    m = m.cuda()
    grid = m.new(sizes[0] * sizes[1], 2)
    srwarp_cuda.projective_grid(m, sizes[0], sizes[1], grid, eps_y, eps_x)
    grid = grid.t().contiguous()
    return grid

@torch.no_grad()
def get_safe(
        grid_raw: torch.Tensor,
        sizes_source: types._II) -> types._TT:

    # (2, 1)
    bound = grid_raw.new_tensor([
        [sizes_source[-1] - 0.5],
        [sizes_source[0] - 0.5]
    ])
    is_in = torch.logical_and(grid_raw >= -0.5, grid_raw < bound)
    is_in = is_in.all(0)
    grid_raw = grid_raw[..., is_in]
    yi, = is_in.nonzero(as_tuple=True)
    return grid_raw, yi

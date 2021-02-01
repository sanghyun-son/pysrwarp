import typing

import torch

from srwarp import wtypes
import srwarp_cuda

@torch.no_grad()
def projective_grid(
        m: torch.Tensor,
        sizes: wtypes._II,
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
    args = [m, sizes[0], sizes[1], grid, eps_y, eps_x]
    if m.dtype == torch.float64:
        srwarp_cuda.projective_grid_double(*args)
    else:
        srwarp_cuda.projective_grid(*args)

    grid = grid.t().contiguous()
    return grid

@torch.no_grad()
def get_safe(
        grid_raw: torch.Tensor,
        sizes_source: wtypes._II) -> wtypes._TT:

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

@torch.no_grad()
def get_safe_projective_grid(
        m: torch.Tensor,
        sizes: wtypes._II,
        sizes_source: wtypes._II,
        eps_y: float=0,
        eps_x: float=0) -> wtypes._TT:

    grid_raw = projective_grid(m, sizes, eps_y=eps_y, eps_x=eps_x)
    grid_raw, yi = get_safe(grid_raw, sizes_source)
    return grid_raw, yi

@torch.no_grad()
def functional_grid(
        f: typing.Callable,
        sizes: wtypes._II,
        scale: typing.Optional[float]=1,
        eps_y: typing.Optional[float]=0,
        eps_x: typing.Optional[float]=0) -> torch.Tensor:

    r = torch.arange(sizes[0] * sizes[1])
    r = r.float()
    r = r.cuda()

    x = r % sizes[1] + eps_x
    y = r // sizes[1] + eps_y
    grid_source = f(x, y)
    grid_source = torch.stack(grid_source, dim=0)
    scale_inv = 1 / scale
    grid_source *= scale_inv
    grid_source += 0.5 * (scale_inv - 1)
    return grid_source

@torch.no_grad()
def get_safe_functional_grid(
        f: typing.Callable,
        sizes: wtypes._II,
        sizes_source: wtypes._II,
        scale: typing.Optional[float]=1,
        eps_y: typing.Optional[float]=0,
        eps_x: typing.Optional[float]=0) -> wtypes._TT:

    grid_raw = functional_grid(f, sizes, scale=scale, eps_y=eps_y, eps_x=eps_x)
    grid_raw, yi = get_safe(grid_raw, sizes_source)
    return grid_raw, yi

@torch.no_grad()
def draw_boundary(
        img: torch.Tensor,
        grid_raw: torch.Tensor,
        yi: torch.Tensor,
        sizes: wtypes._II,
        box_x: int,
        box_y: int,
        box_width: int,
        box_height: int,
        box_thick: int=3) -> torch.Tensor:

    grid_full = torch.zeros(2, sizes[0] * sizes[1])
    grid_full = grid_full.float()
    grid_full = grid_full.cuda()
    grid_full[:, yi] = grid_raw
    grid_full = grid_full.view(2, sizes[0], sizes[1])

    buffer = img.detach().clone()
    def _draw(x: int, y: int) -> None:
        sx, sy = grid_full[:, y, x]
        sx = (sx + 0.5).floor().long().item()
        sy = (sy + 0.5).floor().long().item()

        buffer[:, 0, sy, sx] = 1
        buffer[:, 1, sy, sx] = -1
        buffer[:, 2, sy, sx] = -1
        return

    for i in range(box_width + 2 * box_thick):
        for j in range(box_thick):
            _draw(box_x - box_thick + i, box_y - j)
            _draw(box_x - box_thick + i, box_y + box_height - 1 + j)

    for i in range(box_height + 2 * box_thick):
        for j in range(box_thick):
            _draw(box_x - j, box_y - box_thick + i)
            _draw(box_x + box_width - 1 + j, box_y - box_thick + i)

    return buffer
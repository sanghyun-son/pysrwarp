import math
import typing
import re

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda import amp

from srwarp import wtypes
from srwarp import transform
from srwarp import grid
from srwarp import kernel
from srwarp import svf
from srwarp import adaptive
from srwarp import utils
from srwarp import debug

@torch.no_grad()
def mapping(grid_raw: torch.Tensor, kernel_size: int) -> torch.Tensor:
    '''
    Determine the index of center pixel for filtering.

    Args:
        grid_raw (torch.Tensor): (2, N), where grid[:, i] corresponds to (xi, yi).
        kernel_size (int): The kernel size for filtering.

    Return:
        torch.Tensor: (2, N), indices of center pixels
            where the following kernels are applied on.
            For the even-size kernels, pick a right pixel.
    '''

    if kernel_size % 2 == 0:
        comp = 1
    else:
        comp = 0.5

    # Avoid round()
    grid_discrete = grid_raw + comp
    grid_discrete.floor_()
    grid_discrete = grid_discrete.long()
    return grid_discrete

@torch.no_grad()
def get_local_offset(
        grid_offset: torch.Tensor,
        kernel_size: int,
        padding: typing.Optional[int]=None) -> torch.Tensor:

    '''
    Calculate relative offsets of each point in the kernel support
    with respect to the center pixel.

    Args:
        grid (torch.Tensor): (2, N)

    Return:
        torch.Tensor: (2, N, k)

    '''
    if padding is None:
        padding = kernel_size // 2

    if kernel_size % 2 == 0:
        comp = 0
    else:
        comp = 1

    # (k,)
    local_coord = torch.linspace(
        padding - kernel_size + comp,
        padding + comp - 1,
        kernel_size,
        dtype=grid_offset.dtype,
        device=grid_offset.device,
    )
    # (1, 1, k)
    local_coord = local_coord.view(1, 1, -1)
    # (2, N, 1)
    grid_offset = grid_offset.unsqueeze(-1)
    # (2, N, k)
    local_offset = local_coord - grid_offset
    return local_offset

@amp.autocast(enabled=False)
def warp_by_grid(
        x: torch.Tensor,
        grid_raw: torch.Tensor,
        yi: torch.Tensor,
        sizes: typing.Optional[wtypes._II]=None,
        kernel_type: typing.Union[str, nn.Module]='bicubic',
        padding_type: str='reflect',
        j: typing.Optional[wtypes._TT]=None,
        regularize: bool=True,
        fill: float=-255,
        dump: typing.Optional[dict]=None) -> torch.Tensor:

    if sizes is None:
        sizes = (x.size(-2), x.size(-1))

    if isinstance(kernel_type, str):
        kernels = {'nearest': 1, 'bilinear': 2, 'bicubic': 4}
        if j is None:
            if kernel_type in kernels:
                kernel_size = kernels[kernel_type]
            elif 'gaussian' in kernel_type:
                try:
                    sigma = re.findall('[0-9\.]+', kernel_type)
                    sigma = float(sigma[0])
                except:
                    sigma = 1.0

                kernel_size = 2 * math.ceil(3 * sigma)
            else:
                msg = 'kernel type: {} is not supported!'.format(kernel_type)
                raise ValueError(msg)
        else:
            kernel_size = 12
    else:
        if hasattr(kernel_type, 'kernel_size'):
            kernel_size = kernel_type.kernel_size
        else:
            raise ValueError('kernel size is not specified!')

    with torch.no_grad():
        grid_discrete = mapping(grid_raw, kernel_size)
        grid_offset = (grid_raw - grid_discrete).float()
        local_offset = get_local_offset(grid_offset, kernel_size)
        # (N, k) each
        ox, oy = local_offset.unbind(dim=0)
        # (N, 1, k)
        ox = ox.unsqueeze_(1)
        # (N, k, 1)
        oy = oy.unsqueeze_(-1)

        if dump is not None:
            debug.dump_variable(dump, 'ox', ox)
            debug.dump_variable(dump, 'oy', oy)

        if j is not None:
            ox, oy = adaptive.modulation(
                ox, oy, j, regularize=regularize, dump=dump,
            )

    '''
    if ox.dtype == torch.float64 and oy.dtype == torch.float64:
        ox = ox.float()
        oy = oy.float()
    else:
        raise ValueError('Single precision detected!')
    '''
    # Calculate kernel weights
    if isinstance(kernel_type, str):
        # (N, k, k)
        weight = kernel.calculate_contribution(ox, oy, kernel_type=kernel_type)
    else:
        with torch.no_grad():
            if ox.size(1) == 1:
                ox = ox.repeat(1, kernel_size, 1)

            if oy.size(-1) == 1:
                oy = oy.repeat(1, 1, kernel_size)

        weight = kernel_type(ox, oy)

    # Padding
    pad = kernel_size // 2
    if padding_type == 'zero':
        x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
    else:
        x = utils.padding(x, -2, pad, pad, padding_type=padding_type)
        x = utils.padding(x, -1, pad, pad, padding_type=padding_type)

    xi = grid_discrete[0] + x.size(-1) * grid_discrete[1]
    # Warping
    y = svf.svf_forward(x, weight, sizes, kernel_size, xi, yi, fill, False)
    return y

@amp.autocast(enabled=False)
def warp_by_function(
        x: torch.Tensor,
        f: typing.Union[torch.Tensor, typing.Callable],
        f_inverse: bool=True,
        sizes: typing.Union[wtypes._II, str, None]=None,
        kernel_type: typing.Union[str, nn.Module]='bicubic',
        padding_type: str='reflect',
        adaptive_grid: bool=False,
        regularize: bool=True,
        fill: float=-255,
        dump: typing.Optional[dict]=None) -> torch.Tensor:

    if not f_inverse:
        if isinstance(f, torch.Tensor):
            f = transform.inverse_3x3(f)
        else:
            raise ValueError('Does not support forward warping function!')

    if sizes is None:
        sizes = (x.size(-2), x.size(-1))
    elif sizes == 'auto':
        if isinstance(f, torch.Tensor):
            pass

    if isinstance(f, torch.Tensor):
        grid_raw = grid.projective_grid(f, sizes)
        grid_raw, yi = grid.get_safe(grid_raw, (x.size(-2), x.size(-1)))

    if adaptive_grid:
        j = transform.jacobian(f, sizes=sizes, yi=yi)
    else:
        j = None

    if dump is not None:
        debug.dump_variable(dump, 'grid_raw', grid_raw)
        debug.dump_variable(dump, 'yi', yi)
        debug.dump_variable(dump, 'j', j)

    y = warp_by_grid(
        x,
        grid_raw,
        yi,
        sizes=sizes,
        kernel_type=kernel_type,
        padding_type=padding_type,
        j=j,
        regularize=regularize,
        fill=fill,
        dump=dump,
    )
    return y

if __name__ == '__main__':
    torch.set_printoptions(precision=3, linewidth=240)

    x = torch.arange(16).view(1, 1, 4, 4).float().cuda()
    #print(x)

    import transform
    m = transform.scaling(0.5)
    grid_raw = grid.projective_grid(m, (8, 8))
    grid_raw, yi = grid.get_safe(grid_raw, (4, 4))

    y = warp_by_grid(x, grid_raw, yi, sizes=(8, 8))
    #print(y)

    z = warp_by_function(x, m, sizes=(8, 8), adaptive_grid=True, regularize=False)
    print(z)

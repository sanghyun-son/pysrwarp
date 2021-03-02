import typing

import torch
from torch.nn import functional as F

@torch.no_grad()
def mark_box(
        x: torch.Tensor,
        iy: int,
        ix: int,
        patch_size: int,
        box_thick: int=3,
        color: typing.Tuple[int, int, int]=(255, 0, 0)) -> None:

    '''
    For debugging purpose.
    '''
    color = [color[i] / 127.5 - 1 for i in range(len(color))]

    jy = iy + patch_size
    jx = ix + patch_size
    for i, c in enumerate(color):
        x[..., i, iy:(iy + box_thick), ix:jx] = c
        x[..., i, (jy - box_thick):jy, ix:jx] = c
        x[..., i, iy:jy, ix:(ix + box_thick)] = c
        x[..., i, iy:jy, (jx - box_thick):jx] = c

    return

@torch.no_grad()
def draw_outline(
        x: torch.Tensor,
        ignore_value: float,
        check_diagonal: bool=False,
        color: typing.Tuple[int, int, int]=(0, 0, 0),
        thick: int=1) -> torch.Tensor:

    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    if torch.any(x[..., 0, :] != ignore_value):
        pad_top = 1
    if torch.any(x[..., -1, :] != ignore_value):
        pad_bottom = 1
    if torch.any(x[..., 0] != ignore_value):
        pad_left = 1
    if torch.any(x[..., -1] != ignore_value):
        pad_right = 1

    x = F.pad(
        x,
        (pad_left, pad_right, pad_top, pad_bottom),
        'constant',
        ignore_value,
    )

    color = [color[i] / 127.5 - 1 for i in range(len(color))]

    mask = (x != ignore_value).float()
    if check_diagonal:
        kernel = x.new_ones(3, 3)
    else:
        kernel = x.new_tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    if mask.ndim == 3:
        mask = mask.view(-1, 1, mask.size(1), mask.size(2))
    elif mask.ndim == 4:
        mask = mask.view(-1, 1, mask.size(-2), mask.size(-1))

    kernel = kernel.view(1, 1, 3, 3)
    y = F.conv2d(mask, kernel, padding=1)
    boundary = (1 - mask) * (y > 0)
    boundary = boundary[0, 0]

    ret = x.detach().clone()
    outside = (1 - mask[0, 0] - boundary) * ignore_value
    for i, c in enumerate(color):
        if x.ndim == 3:
            inside = mask * x[i] + boundary * c + outside
            ret[i] = inside
        elif x.ndim == 4:
            inside = mask * x[:, i] + boundary * c + outside
            inside = inside[0]
            ret[:, i] = inside

    if thick > 1:
        return draw_outline(
            ret,
            ignore_value,
            check_diagonal=False,
            color=color,
            thick=(thick - 1),
        )

    return ret

'''
A standalone PyTorch implementation for fast and efficient bicubic resampling.
The resulting values are the same to MATLAB function imresize('bicubic').

## Author:      Sanghyun Son
## Email:       sonsang35@gmail.com (primary), thstkdgus35@snu.ac.kr (secondary)
## Version:     1.2.0
## Last update: July 9th, 2020 (KST)

Depencency: torch

Example::
>>> import torch
>>> import core
>>> x = torch.arange(16).float().view(1, 1, 4, 4)
>>> y = core.imresize(x, sizes=(3, 3))
>>> print(y)
tensor([[[[ 0.7506,  2.1004,  3.4503],
          [ 6.1505,  7.5000,  8.8499],
          [11.5497, 12.8996, 14.2494]]]])
'''

import math
import typing

import torch
from torch.nn import functional as F

from srwarp import kernel
from srwarp import utils
from srwarp import wtypes

__all__ = ['imresize'] 

def discrete_kernel(
        kernel_type: str,
        scale: float,
        antialiasing: bool=True) -> torch.Tensor:

    '''
    For downsampling with integer scale only.
    '''
    downsampling_factor = int(1 / scale)
    if kernel_type == 'bicubic':
        kernel_size_orig = 4
    else:
        raise ValueError('Pass!')

    if antialiasing:
        kernel_size = kernel_size_orig * downsampling_factor
    else:
        kernel_size = kernel_size_orig

    if downsampling_factor % 2 == 0:
        a = kernel_size_orig * (0.5 - 1 / (2 * kernel_size))
    else:
        kernel_size -= 1
        a = kernel_size_orig * (0.5 - 1 / (kernel_size + 1))

    with torch.no_grad():
        r = torch.linspace(-a, a, steps=kernel_size)
        k = kernel.cubic_contribution(r).view(-1, 1)
        k = torch.matmul(k, k.t())
        k /= k.sum()

    return k

def get_padding(
        base: torch.Tensor,
        kernel_size: int,
        x_size: int) -> typing.Tuple[int, int, torch.Tensor]:

    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1

    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        base += pad_pre
    else:
        pad_pre = 0

    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0

    return pad_pre, pad_post, base

def get_weight(
        dist: torch.Tensor,
        kernel_size: int,
        kernel_type: str='bicubic',
        sigma: float=2.0,
        antialiasing_factor: float=1) -> torch.Tensor:

    buffer_pos = dist.new_zeros(kernel_size, len(dist))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(dist - idx)

    # Expand (downsampling) / Shrink (upsampling) the receptive field.
    buffer_pos *= antialiasing_factor
    if kernel_type == 'bicubic':
        weight = kernel.cubic_contribution(buffer_pos)
    else:
        raise ValueError('{} kernel is not supported!'.format(kernel_type))

    weight /= weight.sum(dim=0, keepdim=True)
    return weight

def reshape_tensor(x: torch.Tensor, dim: int, kernel_size: int) -> torch.Tensor:
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1

    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold

def reshape_input(x: torch.Tensor) -> wtypes._TIIII:
    if x.dim() == 4:
        b, c, h, w = x.size()
    elif x.dim() == 3:
        c, h, w = x.size()
        b = None
    elif x.dim() == 2:
        h, w = x.size()
        b = c = None
    else:
        raise ValueError('{}-dim Tensor is not supported!'.format(x.dim()))

    x = x.view(-1, 1, h, w)
    return x, b, c, h, w

def reshape_output(x: torch.Tensor, b: wtypes._I, c: wtypes._I) -> torch.Tensor:
    rh = x.size(-2)
    rw = x.size(-1)
    # Back to the original dimension
    if b is not None:
        x = x.view(b, c, rh, rw)        # 4-dim
    else:
        if c is not None:
            x = x.view(c, rh, rw)       # 3-dim
        else:
            x = x.view(rh, rw)          # 2-dim

    return x

def resize_1d(
        x: torch.Tensor,
        dim: int,
        size: wtypes._I,
        scale: wtypes._F,
        kernel_type: str='bicubic',
        sigma: float=2.0,
        padding_type: str='reflect',
        antialiasing: bool=True) -> torch.Tensor:

    '''
    Args:
        x (torch.Tensor): A torch.Tensor of dimension (B x C, 1, H, W).
        dim (int):
        scale (float):
        size (int):

    Return:
    '''
    # Identity case
    if scale == 1:
        return x

    # Default bicubic kernel with antialiasing (only when downsampling)
    if kernel_type == 'bicubic':
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)

    if antialiasing and (scale < 1):
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sizes
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        pos = torch.linspace(
            0, size - 1, steps=size, dtype=x.dtype, device=x.device,
        )
        pos = (pos + 0.5) / scale - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base
        weight = get_weight(
            dist,
            kernel_size,
            kernel_type=kernel_type,
            sigma=sigma,
            antialiasing_factor=antialiasing_factor,
        )
        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))

    # To backpropagate through x
    x_pad = utils.padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)
    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    x = sample * weight
    x = x.sum(dim=1, keepdim=True)
    return x

def downsampling_2d(
        x: torch.Tensor,
        k: torch.Tensor,
        scale: int,
        padding_type: str='reflect') -> torch.Tensor:

    c = x.size(1)
    k_h = k.size(-2)
    k_w = k.size(-1)

    k = k.to(dtype=x.dtype, device=x.device)
    k = k.view(1, 1, k_h, k_w)
    k = k.repeat(c, c, 1, 1)
    e = torch.eye(c, dtype=k.dtype, device=k.device, requires_grad=False)
    e = e.view(c, c, 1, 1)
    k = k * e

    pad_h = (k_h - scale) // 2
    pad_w = (k_w - scale) // 2
    x = utils.padding(x, -2, pad_h, pad_h, padding_type=padding_type)
    x = utils.padding(x, -1, pad_w, pad_w, padding_type=padding_type)
    y = F.conv2d(x, k, padding=0, stride=scale)
    return y

def imresize(
        x: torch.Tensor,
        scale: wtypes._F=None,
        sizes: typing.Optional[wtypes._II]=None,
        kernel_type: typing.Union[str, torch.Tensor]='bicubic',
        sigma: float=2,
        rotation_degree: float=0,
        padding_type: str='reflect',
        antialiasing: bool=True,
        range_8bit: bool=False) -> torch.Tensor:

    '''
    Args:
        x (torch.Tensor):
        scale (float):
        sizes (tuple(int, int)):
        kernel_type (str, default='cubic'):
        sigma (float, default=2):
        rotation_degree (float, default=0):
        padding_type (str, default='reflect'):
        antialiasing (bool, default=True):

    Return:
        torch.Tensor:
    '''

    if scale is None and sizes is None:
        raise ValueError('One of scale or sizes must be specified!')
    if scale is not None and sizes is not None:
        raise ValueError('Please specify scale or sizes to avoid conflict!')

    x, b, c, h, w = reshape_input(x)

    if sizes is None:
        # Determine output size
        sizes = (math.ceil(h * scale), math.ceil(w * scale))
        scales = (scale, scale)

    if scale is None:
        scales = (sizes[0] / h, sizes[1] / w)

    x, dtype = utils.cast_input(x)

    # Shared keyword arguments across dimensions
    kwargs = {
        'kernel_type': kernel_type,
        'sigma': sigma,
        'padding_type': padding_type,
        'antialiasing': antialiasing,
    }
    # Core resizing module
    if scales[0] < scales[1]:
        order = ((-2, 0), (-1, 1))
    else:
        order = ((-1, 1), (-2, 0))

    for dim, idx in order:
        x = resize_1d(x, dim, size=sizes[idx], scale=scales[idx], **kwargs)
        if range_8bit:
            x = (x + 0.5).floor()
            x = x.clamp(min=0, max=255)

    x = reshape_output(x, b, c)
    x = utils.cast_output(x, dtype)
    return x

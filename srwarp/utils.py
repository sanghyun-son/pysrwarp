import typing

import torch

from srwarp import wtypes

def cast_input(x: torch.Tensor) -> wtypes._TD:
    if not x.dtype.is_floating_point:
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None

    return x, dtype

def cast_output(x: torch.Tensor, dtype: wtypes._D) -> torch.Tensor:
    if dtype is not None:
        if not dtype.is_floating_point:
            x = x.round()
        # To prevent over/underflow when converting types
        if dtype is torch.uint8:
            x = x.clamp(0, 255)

        x = x.to(dtype=dtype)

    return x

def reflect_padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int) -> torch.Tensor:

    '''
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.

    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    '''
    b, c, h, w = x.size()
    if dim == 2 or dim == -2:
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:(h + pad_pre), :].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])
    else:
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:(w + pad_pre)].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])

    return padding_buffer

def padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int,
        padding_type: typing.Optional[str]='reflect') -> torch.Tensor:

    if padding_type is None:
        return x
    elif padding_type == 'reflect':
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))

    x_pad = x_pad.contiguous()
    return x_pad

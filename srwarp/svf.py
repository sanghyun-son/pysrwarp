import typing

import torch
from torch import autograd
from torch.autograd import function

from srwarp import wtypes
import srwarp_cuda


def check_cuda(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError('{} must be on a GPU!'.format(name))

    return

def check_args(
        x: torch.Tensor,
        weight: torch.Tensor,
        k: int,
        xi: torch.Tensor,
        yi: torch.Tensor) -> None:

    check_cuda(x, 'x')
    check_cuda(weight, 'weight')
    check_cuda(xi, 'xi')
    check_cuda(yi, 'yi')

    msg = None
    if x.dim() != 4:
        msg = 'x should be 4-dim Tensor! (got {})'.format(x.dim())

    wk = weight.size(-1)
    if wk != k**2:
        msg = 'Incorrect kernel size! (Expected {}, got {})'.format(k**2, wk)

    if weight.dim() == 3:
        wc = weight.size(1)
        xc = x.size(1)
        if wc != 1 and wc != xc:
            msg = 'Incorrect weight channels! (Expected 1 or {}, got {})'
            msg = msg.format(xc, wc)

    if msg is not None:
        raise ValueError(msg)

    return


class SVF(autograd.Function):

    @staticmethod
    def forward(
            ctx: function._ContextMethodMixin,
            x: torch.Tensor,
            weight: torch.Tensor,
            sizes: typing.Tuple[int, int],
            k: int,
            xi: torch.Tensor,
            yi: torch.Tensor,
            fill_value: float,
            is_half: bool) -> torch.Tensor:

        check_args(x, weight, k, xi, yi)
        if is_half:
            x = x.half()
            weight = weight.half()

        xi = xi.int()
        yi = yi.int()
        # Backup for backpropagation
        ctx.save_for_backward(x, weight, xi, yi)
        ctx.k = k
        ctx.is_half = is_half
        # Return memory allocation
        y = x.new_full((x.size(0), x.size(1), *sizes), fill_value)
        srwarp_cuda.forward(x, weight, y, k, xi, yi, is_half)
        return y

    @staticmethod
    def backward(
            ctx: function._ContextMethodMixin,
            grad_output: torch.Tensor) -> typing.List[wtypes._T]:

        x, weight, xi, yi = ctx.saved_tensors
        k = ctx.k
        is_half = ctx.is_half
        if is_half:
            grad_output = grad_output.half()

        # Return memory allocation
        dx = torch.zeros_like(x)
        dweight = torch.zeros_like(weight)
        srwarp_cuda.backward(
            x, dx, weight, dweight, grad_output, k, xi, yi, is_half,
        )
        return dx, dweight, None, None, None, None, None, None, None, None, None


svf_forward = SVF.apply


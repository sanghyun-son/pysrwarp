import typing

import torch

__all__ = [
    'cubic_contribution',
    'cubic_contribution2d',
]

@torch.no_grad()
def cubic_contribution(x: torch.Tensor, a: float=-0.5) -> torch.Tensor:
    '''
    Apply a cubic spline function to the given offset tensor.

    Args:
        x (torch.Tensor): A offset tensor.
        a (float): Hyperparameter for the cubic spline.

    Return:
        torch.Tensor: f(x), where f is the cubic spline function.
    '''

    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = ax.le(1)
    range_12 = torch.logical_and(ax.gt(1), ax.le(2))

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=x.dtype)

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=x.dtype)

    cont = cont_01 + cont_12
    return cont

@torch.no_grad()
def cubic_contribution2d(
        x: torch.Tensor,
        y: torch.Tensor,
        a: float=-0.5) -> torch.Tensor:

    '''
    Apply a cubic spline function to the given 2D offset vector.

    Args:
        x (torch.Tensor): A offset tensor for x direction.
        y (torch.Tensor): A offset tensor for y direction.
        a (float): Hyperparameter for the cubic spline.

    Return:
        torch.Tensor: f(x) * f(y), where f is the cubic spline function.
    '''

    cont = cubic_contribution(x) * cubic_contribution(y)
    return cont

@torch.no_grad()
def calculate_contribution(
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_type: str='bicubic') -> torch.Tensor:

    if kernel_type == 'nearest':
        return None
    elif kernel_type == 'bilinear':
        return None
    elif kernel_type == 'bicubic':
        weight = cubic_contribution2d(x, y)

    weight = weight.view(weight.size(0), -1)
    weight = weight / weight.sum(-1, keepdim=True)
    return weight

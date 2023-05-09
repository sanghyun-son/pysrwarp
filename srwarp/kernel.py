import typing
import re

import torch

__all__ = [
    'gaussian_contribution',
    'gaussian_contribution_2d',
    'cubic_contribution',
    'cubic_contribution2d',
]

@torch.no_grad()
def gaussian_contribution(x: torch.Tensor, sigma: float=1.0) -> torch.Tensor:
    range_3sigma = (x.abs() <= 3 * sigma + 1)
    # Normalization will be done after
    cont = torch.exp(-x.pow(2) / (2 * sigma**2))
    cont = cont * range_3sigma.to(dtype=x.dtype)
    return cont

@torch.no_grad()
def gaussian_contribution_2d(x: torch.Tensor, y: torch.Tensor, sigma: float=1.0) -> torch.Tensor:
    cont_x = gaussian_contribution(x, sigma=sigma)
    cont_y = gaussian_contribution(y, sigma=sigma)
    cont = cont_x * cont_y
    return cont

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
    elif 'gaussian' in kernel_type:
        try:
            sigma = re.findall('[0-9\.]+', kernel_type)
            sigma = float(sigma[0])
        except:
            sigma = 1.0

        weight = gaussian_contribution_2d(x, y, sigma=sigma)

    weight = weight.view(weight.size(0), -1)
    weight = weight / weight.sum(-1, keepdim=True)
    return weight

import random
import typing

import torch
from torch.nn import functional as F

from srwarp import wtypes
from srwarp import visualization

def crop_preprocess(
        x: torch.Tensor,
        ignore_value: float,
        pool_size: int=4) -> torch.Tensor:

    '''
    Works with convex regions.
    '''
    ignore = (x == ignore_value).float()
    # Shared across batches and channels
    if ignore.dim() == 3:
        ignore = ignore[0]
    elif ignore.dim() == 4:
        ignore = ignore[0, 0]
    elif ignore.dim() > 4:
        msg = 'Invalid dimension! Expected <= 4, Got {}'.format(ignore.dim())
        raise ValueError(msg)

    h, w = ignore.size()
    hh = pool_size * (h // pool_size)
    ww = pool_size * (w // pool_size)
    ignore = ignore[:hh, :ww]
    ignore.unsqueeze_(0).unsqueeze_(0)
    pool = F.max_pool2d(ignore, pool_size)
    return pool

@torch.no_grad()
def crop_search(
        pool: torch.Tensor,
        pool_size: int=4,
        max_level: int=8,
        min_level: int=2) -> typing.Tuple[wtypes._T, wtypes._I]:

    for level in range(max_level, min_level - 1, -1):
        ppool = F.max_pool2d(pool, level, stride=1)
        pos = (ppool == 0).nonzero(as_tuple=False)

        if pos.nelement() > 0:
            return pos, level

    raise ValueError('Cannot find a valid crop!')
    #return None, None

def valid_crop(
        x: torch.Tensor,
        ignore_value: float,
        patch_max: int=256,
        stochastic: bool=True,
        pool_size: int=4,
        margin: int=2,
        debug: bool=False,
        box_thick: int=3) -> typing.Tuple[torch.Tensor, int, int]:

    '''
        x (torch.Tensor): An input image.
        ignore_value (float):
            A value that should not be included in the cropped patch.
        patch_max (int):  The maximum patch size to be cropped.
        stochastic (bool): Add randomness to the cropping positions.
        pool_size (int, optional): Larger pool size yields faster execution
            by sacrificing the number of effective patches.
        margin (int, optional):
            Set margins around the patch boundary to avoid boundary effects.
        debug (bool, optional):
    '''

    pool = crop_preprocess(x, ignore_value, pool_size=pool_size)
    patch_max_wm = patch_max + 2 * margin
    max_level = min(pool.size(-2), pool.size(-1), 1 + patch_max_wm // pool_size)
    pos, level = crop_search(pool, pool_size=pool_size, max_level=max_level)

    if stochastic:
        pos = random.choice(pos)
    else:
        pos = pos[0]

    iy = pool_size * pos[-2].item()
    ix = pool_size * pos[-1].item()
    patch_size = pool_size * level

    if margin > 0:
        iy += margin
        ix += margin
        patch_size -= (2 * margin)

    if patch_max < patch_size:
        iy += random.randrange(0, patch_size - patch_max + 1)
        ix += random.randrange(0, patch_size - patch_max + 1)
        patch_size = patch_max

    patch = x[..., iy:(iy + patch_size), ix:(ix + patch_size)]
    if debug:
        visualization.mark_box(x, iy, ix, patch_size, box_thick=box_thick)

    return patch, iy, ix

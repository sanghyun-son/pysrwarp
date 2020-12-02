import typing

import torch

_I = typing.Optional[int]
_II = typing.Tuple[int, int]
_III = typing.Tuple[int, int]

_F = typing.Optional[float]

_T = typing.Optional[torch.Tensor]
_TT = typing.Tuple[torch.Tensor, torch.Tensor]
_TTTT = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

_D = typing.Optional[torch.dtype]

_LT = typing.List[torch.Tensor]

_TIIII = typing.Tuple[torch.Tensor, _I, _I, _I, _I]
_TD = typing.Tuple[torch.Tensor, _D]
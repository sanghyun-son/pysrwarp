import torch

from srwarp import wtypes

@torch.no_grad()
def dump_variable(dump: dict, key: str, value: object) -> None:
    if value is None:
        dump[key] = None
    if isinstance(value, torch.Tensor):
        dump[key] = value.detach().clone()
    elif isinstance(value, tuple):
        dump[key] = [v.detach().clone() for v in value]

    return

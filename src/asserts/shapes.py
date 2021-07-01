from typing import Tuple, Union

import torch


def same_or_broadcastable(one: Union[torch.Tensor, Tuple], other: Union[torch.Tensor, Tuple]) -> bool:
    if isinstance(one, torch.Tensor):
        one = one.shape

    if isinstance(other, torch.Tensor):
        other = other.shape

    assert isinstance(one, tuple) and isinstance(
        other, tuple
    ), f"Expected tuple input but got {type(one)} and {type(other)}"
    if one == other:
        return True

    if one[-1] == 1 and one[:-1] == other[:-1]:
        return True

    if other[-1] == 1 and one[:-1] == other[:-1]:
        return True

    return False

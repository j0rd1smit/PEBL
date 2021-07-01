import torch


def huber_loss(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return torch.where(torch.abs(x) < delta, torch.pow(x, 2.0) * 0.5, delta * (torch.abs(x) - 0.5 * delta))
import torch


class Bootstrapping:
    MASK = "BootstrappingMASK"


def create_bootstrapping_mask(n_datapoints: int, n_heads: int, bootstrap_prop: float):
    return (torch.rand(n_datapoints, n_heads) < bootstrap_prop).float()

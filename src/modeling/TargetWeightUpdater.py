from typing import Dict

import torch


class TargetWeightUpdater:
    def __init__(
        self,
        *,
        network: torch.nn.Module,
        target: torch.nn.Module,
        sync_rate: int,
        tau: float,
    ) -> None:
        self.network = network
        self.target = target
        self.sync_rate = sync_rate
        self.tau = tau
        self.global_step = 0

    def update_if_needed(self) -> None:
        if self.global_step % self.sync_rate == 0:
            if isinstance(self.target, torch.nn.Module):
                update_state_dict(self.target, self.network.state_dict(), tau=self.tau)
            elif isinstance(self.target, torch.nn.parameter.Parameter):
                update_parameter(self.target, self.network, self.tau)
            else:
                raise Exception(f"Unknown type: {type(self.target)}")

        self.global_step += 1


def update_parameter(target: torch.nn.parameter.Parameter, src: torch.nn.parameter.Parameter, tau: int) -> None:
    with torch.no_grad():
        if tau == 1:
            target.copy_(src)
        else:
            updated_v = tau * src + (1 - tau) * target
            target.copy_(updated_v)


def update_state_dict(model: torch.nn.Module, state_dict: Dict, tau: float = 1) -> None:
    if tau == 1:
        model.load_state_dict(state_dict)
    elif 0 < tau < 1:
        update_sd = {k: tau * state_dict[k] + (1 - tau) * v for k, v in model.state_dict().items()}
        model.load_state_dict(update_sd)

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.Module):
        for p in model.parameters():
            p.requires_grad = False
    elif isinstance(model, torch.nn.parameter.Parameter):
        model.requires_grad = False
    else:
        raise Exception(f"Unknown type: {type(model)}")

    return model


def clip_grad_if_need(parameters: Any, grad_norm_max: float):
    if grad_norm_max > 0:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_norm_max)

    return None


MIN_LOG_NN_OUTPUT = -20
MAX_LOG_NN_OUTPUT = 2
SMALL_NUMBER = 1e-6


def squashed_gaussian(
    inputs: torch.Tensor,
    *,
    deterministic: bool,
    with_logprob: bool,
    action_limit: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    mean, log_std = torch.chunk(inputs, 2, dim=-1)

    log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
    std = torch.exp(log_std)
    pi_distribution = torch.distributions.normal.Normal(mean, std)

    if deterministic:
        # Only used for evaluating policy at test time.
        pi_action = mean
    else:
        pi_action = pi_distribution.rsample()

    if with_logprob:
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
    else:
        logp_pi = None

    pi_action = torch.tanh(pi_action)
    pi_action = action_limit * pi_action

    return pi_action, logp_pi

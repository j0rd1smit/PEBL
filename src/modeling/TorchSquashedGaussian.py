import torch

MIN_LOG_NN_OUTPUT = -20
MAX_LOG_NN_OUTPUT = 2
SMALL_NUMBER = 1e-6

"""
Adapted code from RLLib
https://github.com/ray-project/ray/blob/b01b0f80aa33fc10569f3ab36676ef71fc624d08/rllib/models/torch/torch_action_dist.py#L187
"""


class TorchSquashedGaussian:
    """A tanh-squashed Gaussian distribution defined by: mean, std, low, high.
    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """

    def __init__(self, inputs: torch.Tensor, *, low: float, high: float):
        """Parameterizes the distribution via `inputs`.
        Args:
            low (float): The lowest possible sampling value
                (excluding this value).
            high (float): The highest possible sampling value
                (excluding this value).
        """

        # Split inputs into mean and log(std).
        self.mean, log_std = torch.chunk(inputs, 2, dim=-1)
        # Clip `scale` values (coming from NN) to reasonable values.
        log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        self.std = torch.exp(log_std)
        self.dist = torch.distributions.normal.Normal(self.mean, self.std)
        assert torch.all(low < high)
        self.low = low
        self.high = high

    def deterministic_sample(self) -> torch.Tensor:
        return self._squash(self.dist.mean)

    def rsample(self) -> torch.Tensor:
        # Use the reparameterization version of `dist.sample` to allow for
        # the results to be backprop'able e.g. in a loss term.
        normal_sample = self.dist.rsample()
        return self._squash(normal_sample)

    def logp(self, x: torch.Tensor) -> torch.Tensor:
        # Unsquash values (from [low,high] to ]-inf,inf[)
        unsquashed_values = self._unsquash(x)
        # Get log prob of unsquashed values from our Normal.
        log_prob_gaussian = self.dist.log_prob(unsquashed_values)
        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = torch.clamp(log_prob_gaussian, -100, 100)
        log_prob_gaussian = torch.sum(log_prob_gaussian, dim=-1)
        # Get log-prob for squashed Gaussian.
        unsquashed_values_tanhd = torch.tanh(unsquashed_values)
        log_prob = log_prob_gaussian - torch.sum(torch.log(1 - unsquashed_values_tanhd ** 2 + SMALL_NUMBER), dim=-1)
        return log_prob

    def _squash(self, raw_values: torch.Tensor) -> torch.Tensor:
        # Returned values are within [low, high] (including `low` and `high`).
        squashed = ((torch.tanh(raw_values) + 1.0) / 2.0) * (self.high - self.low) + self.low
        return torch.max(torch.min(squashed, self.high), self.low)

    def _unsquash(self, values: torch.Tensor) -> torch.Tensor:
        normed_values = (values - self.low) / (self.high - self.low) * 2.0 - 1.0
        # Stabilize input to atanh.
        save_normed_values = torch.clamp(normed_values, -1.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER)
        unsquashed = atanh(save_normed_values)
        return unsquashed


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log((1 + x) / (1 - x))

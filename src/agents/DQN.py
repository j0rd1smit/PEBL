import copy
from argparse import ArgumentParser
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pytorch_lightning.core.decorators import auto_move_data

from src.asserts.shapes import same_or_broadcastable
from src.environmental.SampleBatch import SampleBatch
from src.modeling.losses import huber_loss
from src.modeling.networks.CNN import CNN
from src.modeling.networks.MLP import MLP
from src.modeling.TargetWeightUpdater import TargetWeightUpdater
from src.modeling.utils import freeze
from torch.optim import Optimizer

_DEFAULTS: Dict[str, Any] = {
    # Optimization
    "lr": 0.00025,
    "gamma": 0.99,
    "sync_rate": 1000,
    "tau": 1,
    # Exploration
    "eps_start": 1.0,
    "eps_frames": 100_000,
    "eps_min": 0.1,
    # Network
    "n_hidden_units": (128, 128),
    "activation": "RELU",
}


class DQN(pl.LightningModule):
    def __init__(
        self,
        observation_shape: Sequence[int],
        n_actions: int,
        is_online: bool,
        # Optimization
        lr: float = _DEFAULTS["lr"],
        sync_rate: int = _DEFAULTS["sync_rate"],
        tau: float = _DEFAULTS["tau"],
        # Exploration
        eps_start: float = _DEFAULTS["eps_start"],
        eps_frames: int = _DEFAULTS["eps_frames"],
        eps_min: float = _DEFAULTS["eps_min"],
        # Network
        n_hidden_units: Sequence[int] = _DEFAULTS["n_hidden_units"],
        filters: Optional[Sequence[int]] = None,
        kernel_sizes: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
        activation: str = _DEFAULTS["activation"],
        **kwargs: Dict[str, Any],
    ):
        assert eps_start >= eps_min
        super().__init__()
        self.save_hyperparameters()

        self.n_actions = n_actions
        self.network = self._create_network()
        self.target = freeze(copy.deepcopy(self.network))

        self.eps = torch.nn.parameter.Parameter(
            torch.tensor([eps_start], dtype=torch.float32), requires_grad=False
        )

        self.target_weight_updater = TargetWeightUpdater(
            network=self.network,
            target=self.target,
            sync_rate=sync_rate,
            tau=tau,
        )

    @property
    def activation_func(self):
        if self.hparams.activation == "ELU":
            return torch.nn.ELU
        if self.hparams.activation == "RELU":
            return torch.nn.ReLU

        raise Exception("Unknown activation", self.hparams.activation)

    def _create_network(self) -> torch.nn.Module:
        if len(self.hparams.observation_shape) > 1:
            cnn = CNN(
                self.hparams.observation_shape[0],
                self.hparams.filters,
                self.hparams.kernel_sizes,
                self.hparams.strides,
                flatten=True,
                activation=self.activation_func,
            )
            output_shape = cnn(
                torch.ones(self.hparams.observation_shape).unsqueeze(0)
            ).shape[-1]
            return torch.nn.Sequential(
                cnn,
                MLP(
                    n_inputs=output_shape,
                    n_hidden_units=self.hparams.n_hidden_units,
                    n_outputs=self.hparams.n_actions,
                    activation=self.activation_func,
                ),
            )

        return MLP(
            n_inputs=self.hparams.observation_shape[-1],
            n_hidden_units=self.hparams.n_hidden_units,
            n_outputs=self.hparams.n_actions,
            activation=self.activation_func,
        )

    @auto_move_data
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        The inference/prediction step used in deployment.
        Can/should be independent to the training_step.
        :param x: Input tensor.
        :return:  Output tensor.
        """
        return self.network(x.float())

    @auto_move_data
    def select_actions(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self(x)

            return torch.argmax(q_values, -1)

    @auto_move_data
    def select_online_actions(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() <= self.eps:
            return torch.randint(low=0, high=self.n_actions, size=x.shape[:1])

        return self.select_actions(x)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        q_values = self.network(batch[SampleBatch.OBSERVATIONS].float())
        q_values = torch.gather(
            q_values, -1, batch[SampleBatch.ACTIONS].unsqueeze(-1)
        ).squeeze(-1)

        with torch.no_grad():
            q_values_next = self.network(batch[SampleBatch.OBSERVATION_NEXTS].float())
            q_values_targets = self.target(batch[SampleBatch.OBSERVATION_NEXTS].float())

        loss, loss_info = self.loss(
            q_values,
            q_values_next,
            q_values_targets,
            batch[SampleBatch.REWARDS],
            batch[SampleBatch.DONES],
        )
        td_error = loss_info["td_error"]

        if self.hparams.is_online:
            with torch.no_grad():
                self.eps.data = torch.clamp(
                    self.eps - 1 / self.hparams.eps_frames, self.hparams.eps_min
                )
                self.log(
                    "DQN/online/eps",
                    self.eps,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                )

        self.log("DQN/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "DQN/td_error/mean",
            torch.mean(td_error),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "DQN/td_error/max",
            torch.max(td_error),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "DQN/td_error/min",
            torch.min(td_error),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "DQN/q_values/mean",
            torch.mean(q_values),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "DQN/q_values/max",
            torch.max(q_values),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "DQN/q_values/min",
            torch.min(q_values),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        self.target_weight_updater.update_if_needed()

        return loss

    def loss(
        self,
        q_values: torch.Tensor,
        q_values_next: torch.Tensor,
        q_values_targets: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            selected_action = torch.argmax(q_values_next, -1, keepdim=True)
            q_next = torch.gather(q_values_targets, -1, selected_action).squeeze(-1)

            assert same_or_broadcastable(
                q_next.shape, rewards.shape
            ), f"{q_next.shape} != {rewards.shape}"
            q_target = rewards + (1.0 - dones) * self.hparams.gamma * q_next
            q_target = q_target.detach()

        assert q_values.shape == q_target.shape
        td_error = q_values - q_target
        loss = huber_loss(td_error).mean()

        return loss, {"td_error": td_error}

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)
        return [optimizer], []

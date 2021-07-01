import copy
import math
from argparse import ArgumentParser
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data

from src.environmental.SampleBatch import SampleBatch
from src.modeling.losses import huber_loss
from src.modeling.networks.CNN import CNN
from src.modeling.networks.MLP import MLP
from src.modeling.networks.SACQNetwork import SACQNetwork
from src.modeling.TargetWeightUpdater import TargetWeightUpdater
from src.modeling.utils import clip_grad_if_need, freeze, squashed_gaussian
from torch.optim import Optimizer

_DEFAULTS: Dict[str, Any] = {
    # Optimization
    "lr": 3e-4,
    "gamma": 0.99,
    "sync_rate": 1,
    "tau": 0.005,
    "init_alpha": 1.0,
    "grad_norm_max": 10,
    # Network
    "n_hidden_units": (256, 256),
}


class SAC(pl.LightningModule):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_actions: int,
        discrete: bool,
        action_limit: Optional[float] = None,
        # Optimization
        lr: float = _DEFAULTS["lr"],
        sync_rate: int = _DEFAULTS["sync_rate"],
        tau: float = _DEFAULTS["tau"],
        init_alpha: Optional[float] = _DEFAULTS["init_alpha"],
        # Network
        n_hidden_units: Sequence[int] = _DEFAULTS["n_hidden_units"],
        filters: Optional[Sequence[int]] = None,
        kernel_sizes: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.policy_network = self._create_pi_network()
        self.q1_network = self._create_q_network()
        self.q2_network = self._create_q_network()

        self.q1_network_target = freeze(copy.deepcopy(self.q1_network))
        self.q2_network_target = freeze(copy.deepcopy(self.q2_network))
        self.target_weight_updater1 = TargetWeightUpdater(
            network=self.q1_network,
            target=self.q1_network_target,
            sync_rate=sync_rate,
            tau=tau,
        )
        self.target_weight_updater2 = TargetWeightUpdater(
            network=self.q2_network,
            target=self.q2_network_target,
            sync_rate=sync_rate,
            tau=tau,
        )

        if discrete:
            self.target_entropy = -float(0.98 * np.log(1.0 / n_actions))
        else:
            self.target_entropy = -float(n_actions)

        self.log_alpha = torch.nn.parameter.Parameter(
            torch.tensor(
                [math.log(init_alpha) if init_alpha is not None else 0],
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

    @property
    def automatic_optimization(self) -> bool:
        return False

    @property
    def uncertainty_weight(self) -> torch.Tensor:
        return self.log_uncertainty_weight.exp()

    @property
    def alpha_detach(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def _create_q_network(self) -> torch.nn.Module:
        output_size = self.hparams.n_actions if self.hparams.discrete else 1
        if len(self.hparams.observation_shape) > 1:
            cnn = CNN(
                self.hparams.observation_shape[0],
                self.hparams.filters,
                self.hparams.kernel_sizes,
                self.hparams.strides,
                flatten=True,
            )
            n_inputs = cnn(
                torch.ones(self.hparams.observation_shape).unsqueeze(0)
            ).shape[-1]

        else:
            cnn = None
            n_inputs = self.hparams.observation_shape[-1]

        if not self.hparams.discrete:
            n_inputs = n_inputs + self.hparams.n_actions

        fc = MLP(
            n_inputs=n_inputs,
            n_hidden_units=self.hparams.n_hidden_units,
            n_outputs=output_size,
        )

        return SACQNetwork(encoder=cnn, fc=fc)

    def _create_pi_network(self) -> torch.nn.Module:
        output_size = (
            self.hparams.n_actions
            if self.hparams.discrete
            else 2 * self.hparams.n_actions
        )

        if len(self.hparams.observation_shape) > 1:
            cnn = CNN(
                self.hparams.observation_shape[0],
                self.hparams.filters,
                self.hparams.kernel_sizes,
                self.hparams.strides,
                flatten=True,
            )
            n_inputs = cnn(
                torch.ones(self.hparams.observation_shape).unsqueeze(0)
            ).shape[-1]

        else:
            cnn = None
            n_inputs = self.hparams.observation_shape[-1]

        fc = MLP(
            n_inputs=n_inputs,
            n_hidden_units=self.hparams.n_hidden_units,
            n_outputs=output_size,
        )

        if cnn is None:
            return fc

        return torch.nn.Sequential(cnn, fc)

    @auto_move_data
    def select_actions(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.policy_network(x.float())

            if self.hparams.discrete:
                log_pi = F.log_softmax(features, dim=-1)
                pi = torch.exp(log_pi)
                return torch.argmax(pi, -1)

            else:
                pi, _ = squashed_gaussian(
                    features,
                    deterministic=True,
                    with_logprob=False,
                    action_limit=self.hparams.action_limit,
                )
                return pi

    @auto_move_data
    def select_online_actions(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.policy_network(x.float())

            if self.hparams.discrete:
                log_pi = F.log_softmax(features, dim=-1)
                pi = torch.exp(log_pi)

                return torch.distributions.Categorical(probs=pi).sample().detach()

            else:
                pi, _ = squashed_gaussian(
                    features,
                    deterministic=False,
                    with_logprob=False,
                    action_limit=self.hparams.action_limit,
                )
                return pi

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        opt_q, opt_pi, opt_alpha = self.optimizers()

        """
        Q-loss
        """
        if self.hparams.discrete:
            q1 = self.q1_network(batch[SampleBatch.OBSERVATIONS].float())
            q2 = self.q2_network(batch[SampleBatch.OBSERVATIONS].float())
        else:
            q1 = self.q1_network(
                batch[SampleBatch.OBSERVATIONS].float(), batch[SampleBatch.ACTIONS]
            ).squeeze(-1)
            q2 = self.q2_network(
                batch[SampleBatch.OBSERVATIONS].float(), batch[SampleBatch.ACTIONS]
            ).squeeze(-1)

        q = torch.min(q1, q2)

        with torch.no_grad():
            pi_features_next = self.policy_network(
                batch[SampleBatch.OBSERVATION_NEXTS].float()
            )

            if self.hparams.discrete:
                log_pi_next = F.log_softmax(pi_features_next, dim=-1)
                pi_next = torch.exp(log_pi_next)

                entropy_penalty = -self.alpha_detach * log_pi_next

                q1_next = self.q1_network_target(
                    batch[SampleBatch.OBSERVATION_NEXTS].float()
                )
                q2_next = self.q2_network_target(
                    batch[SampleBatch.OBSERVATION_NEXTS].float()
                )

            else:
                pi_next, log_pi_next = squashed_gaussian(
                    pi_features_next,
                    deterministic=False,
                    with_logprob=True,
                    action_limit=self.hparams.action_limit,
                )

                entropy_penalty = torch.unsqueeze(-self.alpha_detach * log_pi_next, -1)

                q1_next = self.q1_network_target(
                    batch[SampleBatch.OBSERVATION_NEXTS], pi_next
                )
                q2_next = self.q2_network_target(
                    batch[SampleBatch.OBSERVATION_NEXTS], pi_next
                )

            q_next = torch.min(q1_next, q2_next)
            assert (
                q_next.shape == entropy_penalty.shape
            ), f"{q_next.shape} != {entropy_penalty.shape}"
            q_next += entropy_penalty

            if self.hparams.discrete:
                q_next = torch.sum(pi_next * q_next, dim=-1)
            else:
                q_next = torch.squeeze(q_next, -1)

            assert (
                batch[SampleBatch.REWARDS].shape == q_next.shape
            ), f"{batch[SampleBatch.REWARDS].shape} != {q_next.shape}"
            assert (
                batch[SampleBatch.DONES].shape == q_next.shape
            ), f"{batch[SampleBatch.DONES].shape} != {q_next.shape}"
            q_target = (
                batch[SampleBatch.REWARDS]
                + (1.0 - batch[SampleBatch.DONES]) * self.hparams.gamma * q_next
            )
            q_target = q_target.detach()

        if self.hparams.discrete:
            q1_action = q1.gather(-1, batch[SampleBatch.ACTIONS].unsqueeze(-1)).squeeze(
                -1
            )
            q2_action = q2.gather(-1, batch[SampleBatch.ACTIONS].unsqueeze(-1)).squeeze(
                -1
            )

            assert (
                q1_action.shape == q_target.shape
            ), f"{q1_action.shape} != {q_target.shape}"
            assert (
                q2_action.shape == q_target.shape
            ), f"{q2_action.shape} != {q_target.shape}"
            td_error1 = q1_action - q_target
            td_error2 = q2_action - q_target
        else:
            assert (
                q1.shape == q_next.shape
            ), f"{torch.squeeze(q1).shape} != {q_target.shape}"
            assert (
                q2.shape == q_next.shape
            ), f"{torch.squeeze(q2).shape} != {q_target.shape}"

            td_error1 = q1 - q_target
            td_error2 = q2 - q_target

        td_error = torch.mean(0.5 * td_error1 + 0.5 * td_error2)

        loss_q1 = torch.mean(huber_loss(td_error1))
        loss_q2 = torch.mean(huber_loss(td_error2))
        loss_q = 0.5 * loss_q1 + 0.5 * loss_q2

        """
        Policy and Alpha loss
        """
        pi_features = self.policy_network(batch[SampleBatch.OBSERVATIONS].float())
        if self.hparams.discrete:
            log_pi = F.log_softmax(pi_features, dim=-1)
            pi = torch.exp(log_pi)
        else:
            pi, log_pi = squashed_gaussian(
                pi_features,
                deterministic=False,
                with_logprob=True,
                action_limit=self.hparams.action_limit,
            )

        # alpha loss
        if self.hparams.discrete:
            assert pi.shape == log_pi.shape, f"{pi.shape} != {log_pi.shape}"
            policy_entropy = -torch.sum(pi * log_pi, -1)
            entropy_target_gap = self.target_entropy - policy_entropy

            loss_alpha = -torch.mean(self.log_alpha * entropy_target_gap.detach())

            self.log(
                "SAC/policy/entropy",
                torch.mean(policy_entropy),
                on_step=False,
                on_epoch=True,
            )
        else:
            entropy_target_gap = log_pi + self.target_entropy
            loss_alpha = self.log_alpha * entropy_target_gap.detach()
            assert len(loss_alpha.shape) == 1
            loss_alpha = -torch.mean(loss_alpha)

        # pi loss
        if self.hparams.discrete:
            assert pi.shape == q.shape, f"{pi.shape} != {q.shape}"
            assert log_pi.shape == q.shape, f"{pi.shape} != {q.shape}"

            q_policy = torch.sum(pi * q.detach(), dim=-1)

            loss_pi = torch.mean(
                torch.sum(pi * (self.alpha_detach * log_pi - q.detach()), dim=-1)
            )
        else:
            q1_new_action = self.q1_network(batch[SampleBatch.OBSERVATIONS], pi)
            q2_new_action = self.q2_network(batch[SampleBatch.OBSERVATIONS], pi)
            q_policy = torch.min(q1_new_action, q2_new_action).squeeze(-1)

            assert log_pi.shape == q_policy.shape, f"{log_pi.shape} != {q_policy.shape}"
            loss_pi = torch.mean((self.alpha_detach * log_pi) - q_policy)

        alpha_grad_norm = self._opt_step(
            opt_alpha, self.log_alpha, loss_alpha, retain_graph=True
        )
        pi_grad_norm = self._opt_step(
            opt_pi, self.policy_network.parameters(), loss_pi, retain_graph=True
        )
        q_grad_norm = self._opt_step(
            opt_q,
            list(self.q1_network.parameters()) + list(self.q2_network.parameters()),
            loss_q,
            retain_graph=False,
        )

        self.target_weight_updater1.update_if_needed()
        self.target_weight_updater2.update_if_needed()

        """
        Logging
        """
        self.log("SAC/alpha", self.alpha_detach, on_step=False, on_epoch=True)
        self.log(
            "SAC/alpha/entropy_target_gap",
            torch.mean(entropy_target_gap),
            on_step=False,
            on_epoch=True,
        )

        self.log("SAC/q/dataset/mean", torch.mean(q), on_step=False, on_epoch=True)
        self.log("SAC/q/dataset/min", torch.min(q), on_step=False, on_epoch=True)
        self.log("SAC/q/dataset/max", torch.max(q), on_step=False, on_epoch=True)

        self.log(
            "SAC/q/policy/mean", torch.mean(q_policy), on_step=False, on_epoch=True
        )
        self.log("SAC/q/policy/min", torch.min(q_policy), on_step=False, on_epoch=True)
        self.log("SAC/q/policy/max", torch.max(q_policy), on_step=False, on_epoch=True)

        self.log(
            "SAC/q/dataset_vs_policy",
            torch.mean(q) - torch.mean(q_policy),
            on_step=False,
            on_epoch=True,
        )

        self.log("SAC/td_error/min", torch.min(td_error), on_step=False, on_epoch=True)
        self.log("SAC/td_error/max", torch.max(td_error), on_step=False, on_epoch=True)
        self.log(
            "SAC/td_error/mean", torch.mean(td_error), on_step=False, on_epoch=True
        )

        self.log("SAC/alpha_loss", loss_alpha, on_step=False, on_epoch=True)
        self.log("SAC/loss_pi", loss_pi, on_step=False, on_epoch=True)
        self.log("SAC/loss_q", loss_q, on_step=False, on_epoch=True)

        if alpha_grad_norm is not None:
            self.log(
                "SAC/grad_norm/alpha", alpha_grad_norm, on_step=False, on_epoch=True
            )
        if pi_grad_norm is not None:
            self.log("SAC/grad_norm/pi", pi_grad_norm, on_step=False, on_epoch=True)
        if q_grad_norm is not None:
            self.log("SAC/grad_norm/q", q_grad_norm, on_step=False, on_epoch=True)

    def _opt_step(self, opt, params, loss, *, retain_graph=False):
        assert torch.isfinite(loss).all(), f"loss = {loss}"
        opt.zero_grad()
        self.manual_backward(loss, opt, retain_graph=retain_graph)
        grad_norm = clip_grad_if_need(params, self.hparams.grad_norm_max)
        opt.step()

        return grad_norm

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self) -> Optimizer:
        opt_q = torch.optim.Adam(
            list(self.q1_network.parameters()) + list(self.q2_network.parameters()),
            lr=self.hparams.lr,
        )
        opt_pi = torch.optim.Adam(self.policy_network.parameters(), lr=self.hparams.lr)
        opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.hparams.lr)
        return [opt_q, opt_pi, opt_alpha], []

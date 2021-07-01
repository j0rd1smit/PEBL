import copy
import math
from argparse import ArgumentParser
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.asserts import same_or_broadcastable
from pytorch_lightning.core.decorators import auto_move_data
from src.datasets.boostrapping import Bootstrapping
from src.environmental.SampleBatch import SampleBatch
from src.modeling.losses import huber_loss
from src.modeling.networks.CNN import CNN
from src.modeling.networks.Ensemble import Ensemble
from src.modeling.networks.EnsembleMLP import EnsembleMLP
from src.modeling.networks.MLP import MLP
from src.modeling.networks.NetworkWithPrior import NetworkWithPrior
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
    "auto_alpha_tuning": True,
    "init_alpha": 1.0,
    "grad_norm_max": 10,
    "target_entropy": None,
    # Uncertainty optimization
    "policy_uncertainty_weight": 1,
    "policy_uncertainty_weight_target": 1,
    "policy_uncertainty_weight_auto_tune": True,
    # Network
    "n_hidden_units": (256, 256),
    # Prior
    "n_heads": 15,
    "prior_weight": 10,
    "conv1d_mlp": True,
}


class PEBLSAC(pl.LightningModule):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_actions: int,
        discrete: bool,
        action_limit: Optional[Sequence[float]] = None,
        # Optimization
        lr: float = _DEFAULTS["lr"],
        sync_rate: int = _DEFAULTS["sync_rate"],
        tau: float = _DEFAULTS["tau"],
        init_alpha: Optional[float] = _DEFAULTS["init_alpha"],
        auto_alpha_tuning: bool = _DEFAULTS["auto_alpha_tuning"],
        grad_norm_max: float = _DEFAULTS["grad_norm_max"],
        target_entropy: Optional[float] = _DEFAULTS["target_entropy"],
        # Uncertainty optimization
        policy_uncertainty_weight_auto_tune: bool = _DEFAULTS[
            "policy_uncertainty_weight_auto_tune"
        ],
        policy_uncertainty_weight: float = _DEFAULTS["policy_uncertainty_weight"],
        policy_uncertainty_weight_target: float = _DEFAULTS[
            "policy_uncertainty_weight_target"
        ],
        # Network
        n_hidden_units: Sequence[int] = _DEFAULTS["n_hidden_units"],
        filters: Optional[Sequence[int]] = None,
        kernel_sizes: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
        # priors
        n_heads: int = _DEFAULTS["n_heads"],
        prior_weight: float = _DEFAULTS["prior_weight"],
        conv1d_mlp: bool = _DEFAULTS["conv1d_mlp"],
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

        if target_entropy is not None:
            self.target_entropy = target_entropy
        else:
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
        self.log_uncertainty_weight = torch.nn.parameter.Parameter(
            torch.tensor([math.log(policy_uncertainty_weight)], dtype=torch.float32),
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

        fc = self._create_q_fc(
            n_inputs=n_inputs,
            n_hidden_units=self.hparams.n_hidden_units,
            n_outputs=output_size,
        )

        return SACQNetwork(encoder=cnn, fc=fc)

    def _create_q_fc(self, n_inputs, n_hidden_units, n_outputs):
        if self.hparams.prior_weight > 0:
            prior = self._create_q_mlp(
                n_inputs=n_inputs,
                n_hidden_units=n_hidden_units,
                n_outputs=n_outputs,
            )
            fc = self._create_q_mlp(
                n_inputs=n_inputs,
                n_hidden_units=n_hidden_units,
                n_outputs=n_outputs,
            )

            return NetworkWithPrior(
                network=fc,
                prior_network=prior,
                prior_scale=self.hparams.prior_weight,
            )

        return self._create_q_mlp(
            n_inputs=n_inputs,
            n_hidden_units=n_hidden_units,
            n_outputs=n_outputs,
        )

    def _create_q_mlp(self, n_inputs, n_hidden_units, n_outputs):
        if self.hparams.conv1d_mlp:
            return EnsembleMLP(
                n_inputs=n_inputs,
                n_hidden_units=n_hidden_units,
                n_outputs=n_outputs,
                n_heads=self.hparams.n_heads,
            )
        else:
            models = [
                MLP(
                    n_inputs=n_inputs,
                    n_hidden_units=n_hidden_units,
                    n_outputs=n_outputs,
                )
                for _ in range(self.hparams.n_heads)
            ]
            return Ensemble(models)

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
                # print(pi)

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
    def agent_info(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        info = {
            Bootstrapping.MASK: (torch.rand([1, self.hparams.n_heads]) < 0.8).float()
        }

        return info

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        opt_q, opt_pi, opt_alpha, uncertainty_opt = self.optimizers()

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
        q_std, q_mean = torch.std_mean(q, 1)

        if self.hparams.discrete:
            q_std_action = torch.gather(
                q_std, -1, torch.unsqueeze(batch[SampleBatch.ACTIONS], -1)
            ).squeeze(-1)
        else:
            q_std_action = q_std

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

                entropy_penalty = -torch.unsqueeze(self.alpha_detach * log_pi_next, -1)

                q1_next = self.q1_network_target(
                    batch[SampleBatch.OBSERVATION_NEXTS], pi_next
                )
                q2_next = self.q2_network_target(
                    batch[SampleBatch.OBSERVATION_NEXTS], pi_next
                )

            q_next = torch.min(q1_next, q2_next)

            q_next_std, q_next_mean = torch.std_mean(q_next, 1)
            q_next = (
                q_next_mean - self.hparams.policy_uncertainty_weight_target * q_next_std
            )

            assert (
                q_next.shape == entropy_penalty.shape
            ), f"{q_next.shape} != {entropy_penalty.shape}"
            q_next += entropy_penalty

            if self.hparams.discrete:
                assert (
                    pi_next.shape == q_next.shape
                ), f"{pi_next.shape} != {q_next.shape}"
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
            q_target = q_target.detach().unsqueeze(-1)

        if self.hparams.discrete:
            actions = torch.repeat_interleave(
                torch.unsqueeze(batch[SampleBatch.ACTIONS], 1), self.hparams.n_heads, 1
            ).unsqueeze(-1)
            q1_action = q1.gather(-1, actions).squeeze(-1)
            q2_action = q2.gather(-1, actions).squeeze(-1)

            assert same_or_broadcastable(
                q1_action.shape, q_target.shape
            ), f"{q1_action.shape} != {q_target.shape}"
            assert same_or_broadcastable(
                q2_action.shape, q_target.shape
            ), f"{q2_action.shape} != {q_target.shape}"
            td_error1 = q1_action - q_target
            td_error2 = q2_action - q_target
        else:
            assert same_or_broadcastable(
                q1.shape, q_target.shape
            ), f"{q1.shape} != {q_target.shape}"
            assert same_or_broadcastable(
                q2.shape, q_target.shape
            ), f"{q2.shape} != {q_target.shape}"

            td_error1 = q1 - q_target
            td_error2 = q2 - q_target

        td_error = torch.mean(0.5 * td_error1 + 0.5 * td_error2)

        loss_q1_per_head = huber_loss(td_error1) * batch[Bootstrapping.MASK]
        loss_q1_per_datapoint = loss_q1_per_head.sum(1) / (
            batch[Bootstrapping.MASK].sum(1) + 1e-7
        )
        loss_q1 = torch.mean(loss_q1_per_datapoint)

        loss_q2_per_head = huber_loss(td_error2) * batch[Bootstrapping.MASK]
        loss_q2_per_datapoint = loss_q2_per_head.sum(1) / (
            batch[Bootstrapping.MASK].sum(1) + 1e-7
        )
        loss_q2 = torch.mean(loss_q2_per_datapoint)

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
        if self.hparams.auto_alpha_tuning:
            if self.hparams.discrete:
                assert pi.shape == log_pi.shape, f"{pi.shape} != {log_pi.shape}"
                policy_entropy = -torch.sum(pi * log_pi, -1)
                entropy_target_gap = self.target_entropy - policy_entropy

                loss_alpha = -torch.mean(self.log_alpha * entropy_target_gap.detach())

                self.log(
                    "PEBLSAC/policy/entropy",
                    torch.mean(policy_entropy),
                    on_step=False,
                    on_epoch=True,
                )
            else:
                entropy_target_gap = log_pi + self.target_entropy
                loss_alpha = self.log_alpha * entropy_target_gap.detach()
                assert len(loss_alpha.shape) == 1
                loss_alpha = -torch.mean(loss_alpha)
        else:
            loss_alpha = None
            entropy_target_gap = None

        # pi loss
        if self.hparams.discrete:
            q_with_penalty = q_mean - self.uncertainty_weight.detach() * q_std
            assert (
                pi.shape == q_with_penalty.shape
            ), f"{pi.shape} != {q_with_penalty.shape}"
            assert (
                log_pi.shape == q_with_penalty.shape
            ), f"{pi.shape} != {q_with_penalty.shape}"

            q_policy = torch.sum(pi * q_with_penalty.detach(), dim=-1)
            q_policy_std = torch.sum(pi * q_std, dim=-1)

            loss_pi = torch.mean(
                torch.sum(
                    pi * ((self.alpha_detach * log_pi) - q_with_penalty.detach()),
                    dim=-1,
                )
            )
        else:
            q1_new_action = self.q1_network(batch[SampleBatch.OBSERVATIONS], pi)
            q2_new_action = self.q2_network(batch[SampleBatch.OBSERVATIONS], pi)
            q_policy = torch.min(q1_new_action, q2_new_action).squeeze(-1)

            q_policy_std, q_policy_mean = torch.std_mean(q_policy, 1)
            q_policy = q_policy_mean - self.uncertainty_weight.detach() * q_policy_std

            assert log_pi.shape == q_policy.shape, f"{log_pi.shape} != {q_policy.shape}"
            loss_pi = torch.mean((self.alpha_detach * log_pi) - q_policy)

        if self.hparams.policy_uncertainty_weight_auto_tune:
            assert (
                q_policy_std.shape == q_std_action.shape
            ), f"{q_policy_std.shape} != {q_std_action.shape}"
            policy_vs_dataset_uncertainty_gap = torch.mean(q_policy_std) - torch.mean(
                q_std_action
            )

            uncertainty_weight_loss = (
                -self.uncertainty_weight * policy_vs_dataset_uncertainty_gap
            )
        else:
            uncertainty_weight_loss = None
            policy_vs_dataset_uncertainty_gap = None

        if uncertainty_weight_loss is not None:
            uncertainty_weight_grad_norm = self._opt_step(
                uncertainty_opt,
                self.log_uncertainty_weight,
                uncertainty_weight_loss,
                retain_graph=True,
            )
        else:
            uncertainty_weight_grad_norm = None

        if self.hparams.auto_alpha_tuning:
            alpha_grad_norm = self._opt_step(
                opt_alpha, self.log_alpha, loss_alpha, retain_graph=True
            )
        else:
            alpha_grad_norm = None
        pi_grad_norm = self._opt_step(
            opt_pi, self.policy_network.parameters(), loss_pi, retain_graph=False
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
        self.log("PEBLSAC/alpha", self.alpha_detach, on_step=False, on_epoch=True)
        if entropy_target_gap is not None:
            self.log(
                "PEBLSAC/alpha/entropy_target_gap",
                torch.mean(entropy_target_gap),
                on_step=False,
                on_epoch=True,
            )
        self.log(
            "PEBLSAC/alpha/entropy_penalty",
            torch.mean(entropy_penalty),
            on_step=False,
            on_epoch=True,
        )

        self.log("PEBLSAC/q/dataset/mean", torch.mean(q), on_step=False, on_epoch=True)
        self.log("PEBLSAC/q/dataset/min", torch.min(q), on_step=False, on_epoch=True)
        self.log("PEBLSAC/q/dataset/max", torch.max(q), on_step=False, on_epoch=True)

        self.log(
            "PEBLSAC/q/policy/mean", torch.mean(q_policy), on_step=False, on_epoch=True
        )
        self.log(
            "PEBLSAC/q/policy/min", torch.min(q_policy), on_step=False, on_epoch=True
        )
        self.log(
            "PEBLSAC/q/policy/max", torch.max(q_policy), on_step=False, on_epoch=True
        )

        self.log(
            "PEBLSAC/q/policy_vs_dataset",
            torch.mean(q_policy) - torch.mean(q),
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "PEBLSAC/uncertainty/dataset/mean",
            torch.mean(q_std),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "PEBLSAC/uncertainty/dataset/min",
            torch.min(q_std),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "PEBLSAC/uncertainty/dataset/max",
            torch.max(q_std),
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "PEBLSAC/uncertainty/policy/mean",
            torch.mean(q_policy_std),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "PEBLSAC/uncertainty/policy/min",
            torch.min(q_policy_std),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "PEBLSAC/uncertainty/policy/max",
            torch.max(q_policy_std),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "PEBLSAC/uncertainty/gap/policy_vs_dataset",
            torch.mean(q_policy_std) - torch.mean(q_std_action),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "PEBLSAC/uncertainty/weight",
            self.uncertainty_weight,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "PEBLSAC/td_error/min", torch.min(td_error), on_step=False, on_epoch=True
        )
        self.log(
            "PEBLSAC/td_error/max", torch.max(td_error), on_step=False, on_epoch=True
        )
        self.log(
            "PEBLSAC/td_error/mean", torch.mean(td_error), on_step=False, on_epoch=True
        )

        if loss_alpha is not None:
            self.log("PEBLSAC/loss/alpha", loss_alpha, on_step=False, on_epoch=True)
        self.log("PEBLSAC/loss/pi", loss_pi, on_step=False, on_epoch=True)
        self.log("PEBLSAC/loss/q", loss_q, on_step=False, on_epoch=True)
        if uncertainty_weight_loss is not None:
            self.log(
                "PEBLSAC/loss/uncertainty_weight",
                uncertainty_weight_loss,
                on_step=False,
                on_epoch=True,
            )

        if alpha_grad_norm is not None:
            self.log(
                "PEBLSAC/grad_norm/alpha", alpha_grad_norm, on_step=False, on_epoch=True
            )
        if pi_grad_norm is not None:
            self.log("PEBLSAC/grad_norm/pi", pi_grad_norm, on_step=False, on_epoch=True)
        if q_grad_norm is not None:
            self.log("PEBLSAC/grad_norm/q", q_grad_norm, on_step=False, on_epoch=True)
        if uncertainty_weight_grad_norm is not None:
            self.log(
                "PEBLSAC/grad_norm/q",
                uncertainty_weight_grad_norm,
                on_step=False,
                on_epoch=True,
            )

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
        opt_lagrange = torch.optim.Adam(
            [self.log_uncertainty_weight], lr=self.hparams.lr
        )
        return [opt_q, opt_pi, opt_alpha, opt_lagrange], []

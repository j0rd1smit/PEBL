import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from src.asserts import same_or_broadcastable
from src.datasets.boostrapping import Bootstrapping
from src.environmental.SampleBatch import SampleBatch
from src.modeling.losses import huber_loss
from src.modeling.networks.CNN import CNN
from src.modeling.networks.Ensemble import Ensemble
from src.modeling.networks.EnsembleMLP import EnsembleMLP
from src.modeling.networks.MLP import MLP
from src.modeling.networks.NetworkWithPrior import NetworkWithPrior
from src.modeling.TargetWeightUpdater import TargetWeightUpdater
from src.modeling.utils import clip_grad_if_need, freeze
from torch.optim import Optimizer

_DEFAULTS: Dict[str, Any] = {
    # Optimization
    "lr": 0.00025,
    "gamma": 0.99,
    "sync_rate": 1000,
    "tau": 1,
    "policy_uncertainty_weight": 1.0,
    "policy_uncertainty_weight_auto_tune": False,
    "grad_norm_max": 10,
    "uncertainty_weight_q_next": False,
    # Network
    "n_hidden_units": (128, 128),
    "prior_weight": 10,
    "conv1d_mlp": True,
    "dropout_probs": 0.0,
    "activation": "ELU",
}


class PEBLDQN(pl.LightningModule):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_actions: int,
        n_heads: int,
        # Optimization
        gamma: float = _DEFAULTS["gamma"],
        lr: float = _DEFAULTS["lr"],
        sync_rate: int = _DEFAULTS["sync_rate"],
        tau: float = _DEFAULTS["tau"],
        grad_norm_max: float = _DEFAULTS["grad_norm_max"],
        uncertainty_weight_q_next: bool = _DEFAULTS["uncertainty_weight_q_next"],
        # Exploration
        # Network
        activation: str = _DEFAULTS["activation"],
        dropout_probs: float = _DEFAULTS["dropout_probs"],
        n_hidden_units: Sequence[int] = _DEFAULTS["n_hidden_units"],
        prior_weight: float = _DEFAULTS["prior_weight"],
        conv1d_mlp: bool = _DEFAULTS["conv1d_mlp"],
        policy_uncertainty_weight: float = _DEFAULTS["policy_uncertainty_weight"],
        policy_uncertainty_weight_auto_tune: bool = _DEFAULTS[
            "policy_uncertainty_weight_auto_tune"
        ],
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.network = self._create_network()
        self.target = freeze(copy.deepcopy(self.network))

        self.target_weight_updater = TargetWeightUpdater(
            network=self.network,
            target=self.target,
            sync_rate=sync_rate,
            tau=tau,
        )

        self.log_uncertainty_weight = torch.nn.parameter.Parameter(
            torch.tensor([math.log(policy_uncertainty_weight)], dtype=torch.float32),
            requires_grad=True,
        )

    @property
    def automatic_optimization(self) -> bool:
        return False

    @property
    def activation_func(self):
        if self.hparams.activation == "ELU":
            return torch.nn.ELU
        if self.hparams.activation == "RELU":
            return torch.nn.ReLU

        raise Exception("Unknown activation", self.hparams.activation)

    @property
    def uncertainty_weight(self) -> torch.Tensor:
        return self.log_uncertainty_weight.exp()

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
            n_inputs = cnn(
                torch.ones(self.hparams.observation_shape).unsqueeze(0)
            ).shape[-1]

        else:
            cnn = None
            n_inputs = self.hparams.observation_shape[-1]

        fc = self.create_mlp(
            n_inputs=n_inputs,
            n_hidden_units=self.hparams.n_hidden_units,
            n_outputs=self.hparams.n_actions,
            dropout_probs=self.hparams.dropout_probs,
        )
        if self.hparams.prior_weight > 0:
            prior = self.create_mlp(
                n_inputs=n_inputs,
                n_hidden_units=self.hparams.n_hidden_units,
                n_outputs=self.hparams.n_actions,
            )
        else:
            prior = None

        return self._combine_network(cnn, fc, prior)

    def _combine_network(
        self,
        cnn: Optional[torch.nn.Module],
        fc: torch.nn.Module,
        prior: Optional[torch.nn.Module],
    ):
        if prior is not None:
            return NetworkWithPrior(
                shared_encoder=cnn,
                network=fc,
                prior_network=prior,
                prior_scale=self.hparams.prior_weight,
            )

        if cnn is None:
            return fc

        return torch.nn.Sequential(cnn, fc)

    def create_mlp(
        self,
        n_inputs: int,
        n_hidden_units: List[int],
        n_outputs: List[int],
        dropout_probs: Union[float, List[float]] = 0.0,
    ):
        if self.hparams.conv1d_mlp:
            return EnsembleMLP(
                n_inputs=n_inputs,
                n_hidden_units=n_hidden_units,
                n_outputs=n_outputs,
                n_heads=self.hparams.n_heads,
                activation=self.activation_func,
                dropout_probs=dropout_probs,
            )
        else:
            models = [
                MLP(
                    n_inputs=n_inputs,
                    n_hidden_units=n_hidden_units,
                    n_outputs=n_outputs,
                    dropout_probs=dropout_probs,
                    activation=self.activation_func,
                )
                for _ in range(self.hparams.n_heads)
            ]
            return Ensemble(models)

    def forward(
        self,
        x,
        *args,
        **kwargs,
    ):
        q_values = self.network(x.float())

        if self.hparams.n_heads > 1:
            q_values_std, q_values = torch.std_mean(q_values, 1)
            return q_values, q_values_std

        return q_values, torch.zeros_like(q_values)

    @auto_move_data
    def select_actions(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values, q_values_std = self(x)
            selected_action = torch.argmax(
                q_values - self.uncertainty_weight * q_values_std, -1
            )
            # most_certaint_action = torch.argmax(-self.uncertainty_weight * q_values_std, -1)

            # print(q_values - self.uncertainty_weight * q_values_std)
            # print(q_values_std)
            # print(q_values_std[0][selected_action])
            # options = ["NoOp", "left", "NoOp", "right", "NoOp", "fire"]
            # print(options[int(selected_action)], options[int(most_certaint_action)])
            # if q_values_std[0][selected_action] > 0.175:
            #    import time

            #    time.sleep(1)

            return selected_action

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        opt_q, opt_lagrange = self.optimizers()

        actions = torch.unsqueeze(batch[SampleBatch.ACTIONS], 1)
        actions_per_head = torch.repeat_interleave(
            actions, self.hparams.n_heads, 1
        ).unsqueeze(-1)
        rewards = torch.repeat_interleave(
            torch.unsqueeze(batch[SampleBatch.REWARDS], 1), self.hparams.n_heads, 1
        )
        dones = torch.repeat_interleave(
            torch.unsqueeze(batch[SampleBatch.DONES], 1), self.hparams.n_heads, 1
        )

        q_values = self.network(batch[SampleBatch.OBSERVATIONS].float())
        q_values_actions = torch.gather(q_values, -1, actions_per_head).squeeze(-1)

        with torch.no_grad():
            q_values_next = self.network(batch[SampleBatch.OBSERVATION_NEXTS].float())
            q_values_targets = self.target(batch[SampleBatch.OBSERVATION_NEXTS].float())

        loss, loss_info = self.loss(
            q_values=q_values_actions,
            q_values_next=q_values_next,
            q_values_targets=q_values_targets,
            rewards=rewards,
            dones=dones,
            bootstrapping_masks=batch[Bootstrapping.MASK],
        )

        q_values_std, q_values_mean = torch.std_mean(q_values, 1)
        penalized_q_values = q_values_mean - self.uncertainty_weight * q_values_std
        policy = torch.argmax(penalized_q_values, 1, keepdim=True)

        if self.hparams.policy_uncertainty_weight_auto_tune:
            lagrange_loss, lagrange_loss_info = self.lagrange_loss(
                q_values_std, policy, actions
            )

            self._opt_step(
                opt_lagrange,
                self.log_uncertainty_weight,
                lagrange_loss,
                retain_graph=True,
            )
        else:
            lagrange_loss = None
            lagrange_loss_info = None

        self._opt_step(opt_q, self.network.parameters(), loss)

        self.target_weight_updater.update_if_needed()

        # #### logging ####
        # Loss
        self.log("PEBLDQN/loss/q", loss, prog_bar=False, on_epoch=True, on_step=False)
        if lagrange_loss is not None:
            self.log(
                "PEBLDQN/loss/uw_loss",
                lagrange_loss,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
            )

        # TD
        self.log(
            "PEBLDQN/td_error",
            torch.mean(loss_info["td_error"]),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/td_error/median",
            torch.median(loss_info["td_error"]),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/td_error/min",
            torch.min(loss_info["td_error"]),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/td_error/max",
            torch.max(loss_info["td_error"]),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        self.log(
            "PEBLDQN/q_values/dataset/mean",
            torch.mean(q_values_actions),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/q_values/dataset/max",
            torch.max(q_values_actions),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/q_values/dataset/min",
            torch.min(q_values_actions),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        # Q values dataset
        q_values_dataset = torch.gather(penalized_q_values, 1, actions)
        self.log(
            "PEBLDQN/q_values/with_penalized/dataset/mean",
            torch.mean(q_values_dataset),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/q_values/with_penalized/dataset/max",
            torch.max(q_values_dataset),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/q_values/with_penalized/dataset/min",
            torch.min(q_values_dataset),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        # Q values policy
        q_values_policy = torch.gather(penalized_q_values, 1, policy)
        self.log(
            "PEBLDQN/q_values/with_penalized/policy/mean",
            torch.mean(q_values_policy),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/q_values/with_penalized/policy/max",
            torch.max(q_values_policy),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/q_values/with_penalized/policy/min",
            torch.min(q_values_policy),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        # Uncertainty
        q_values_dataset_std = torch.gather(q_values_std, 1, actions)
        self.log(
            "PEBLDQN/uncertainty/dataset/max",
            torch.max(q_values_dataset_std),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/uncertainty/dataset/mean",
            torch.mean(q_values_dataset_std),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/uncertainty/dataset/min",
            torch.min(q_values_dataset_std),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        q_values_policy_std = torch.gather(q_values_std, 1, policy)
        self.log(
            "PEBLDQN/uncertainty/policy/max",
            torch.max(q_values_policy_std),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/uncertainty/policy/mean",
            torch.mean(q_values_policy_std),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PEBLDQN/uncertainty/policy/min",
            torch.min(q_values_policy_std),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        policy_vs_dataset = torch.mean(q_values_policy_std) - torch.mean(
            q_values_dataset_std
        )
        self.log(
            "PEBLDQN/uncertainty/policy_vs_dataset",
            policy_vs_dataset,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        if self.hparams.policy_uncertainty_weight_auto_tune:
            self.log(
                "PEBLDQN/uncertainty_weight",
                self.uncertainty_weight,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
            )

    def loss(
        self,
        q_values,
        q_values_next,
        q_values_targets,
        rewards,
        dones,
        bootstrapping_masks,
    ):
        with torch.no_grad():
            q_values_next_std, q_values_next_mean = torch.std_mean(
                q_values_next, 1, keepdim=True
            )
            q_values_next = (
                q_values_next_mean - self.uncertainty_weight * q_values_next_std
            )

            q_values_targets_std, q_values_targets_mean = torch.std_mean(
                q_values_targets, 1, keepdim=True
            )
            if self.hparams.uncertainty_weight_q_next:
                q_values_targets = (
                    q_values_targets_mean
                    - self.uncertainty_weight * q_values_targets_std
                )
            else:
                q_values_targets = q_values_targets_mean - q_values_targets_std

            selected_action = torch.argmax(q_values_next, -1, keepdim=True)
            q_next = torch.gather(q_values_targets, -1, selected_action).squeeze(-1)

            assert same_or_broadcastable(
                q_next.shape, rewards.shape
            ), f"{q_next.shape} != {rewards.shape}"
            q_target = rewards + (1.0 - dones) * self.hparams.gamma * q_next
            q_target = q_target.detach()

        assert same_or_broadcastable(
            q_values.shape, q_target.shape
        ), f"{q_values.shape} != {q_target.shape}"
        td_error = q_values - q_target

        losses_per_head = huber_loss(td_error)
        assert losses_per_head.shape == bootstrapping_masks.shape
        losses_per_head_with_mask = losses_per_head * bootstrapping_masks
        losses_per_datapoint = losses_per_head_with_mask.sum(1) / (
            bootstrapping_masks.sum(1) + 1e-7
        )
        loss = torch.mean(losses_per_datapoint)

        return loss, {
            "td_error": torch.masked_select(td_error, bootstrapping_masks == 1.0),
        }

    def lagrange_loss(
        self,
        q_values_std,
        policy,
        actions,
    ):
        policy_std = torch.gather(q_values_std, -1, policy)
        action_std = torch.gather(q_values_std, -1, actions)
        gap = torch.mean(policy_std) - torch.mean(action_std)

        loss = -self.uncertainty_weight * gap

        return loss, {
            "gap": gap,
        }

    def _opt_step(self, opt, params, loss, *, retain_graph=False):
        assert torch.isfinite(loss).all(), f"loss = {loss}"
        opt.zero_grad()
        self.manual_backward(loss, opt, retain_graph=retain_graph)
        clip_grad_if_need(params, self.hparams.grad_norm_max)
        opt.step()

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)  # type: ignore
        opt_lagrange = torch.optim.Adam([self.log_uncertainty_weight], lr=self.hparams.lr)  # type: ignore
        return [optimizer, opt_lagrange], []

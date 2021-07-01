import copy
import pprint

import gym
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from src.agents.PEBLSAC import PEBLSAC
from src.builders import eval_callback
from src.datasets.D4RLDataModule import D4RLDataModule
from src.utils.logging import (create_return_based_model_checkpoint,
                               create_tb_and_csv_logger, max_epochs)


def main() -> None:
    from argparse import ArgumentParser

    pp = pprint.PrettyPrinter(indent=2)
    parser = ArgumentParser()

    parser.add_argument("--gpu", default=True, type=bool)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    parser.add_argument("--max_steps", default=500_000, type=int)
    parser.add_argument(
        "--env_name",
        type=str,
        default="halfcheetah-medium-v2",
        choices=[
            "maze2d-open-v0",
            "maze2d-umaze-v1",
            "maze2d-medium-v1",
            "maze2d-large-v1",
        ]
        + d4rl_gym_datasets(),
    )

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    seed_everything(args.seed)

    env = gym.make(args.env_name)
    action_space = env.action_space
    discrete = isinstance(action_space, gym.spaces.Discrete)
    if discrete:
        n_actions = int(action_space.n)
        action_limit = None
    else:
        n_actions = int(np.prod(action_space.shape))
        assert np.all(env.action_space.high == env.action_space.high[0])
        action_limit = float(env.action_space.high[0])

    config = env_defaults(args)
    hparams = {**config, **copy.deepcopy(args.__dict__)}
    pp.pprint(hparams)

    model = PEBLSAC(
        observation_shape=env.observation_space.shape,
        n_actions=n_actions,
        discrete=discrete,
        action_limit=action_limit,
        **hparams,
    )

    data_module = D4RLDataModule(
        env_name=hparams["env_name"],
        batch_size=hparams["batch_size"],
        steps_per_epoch=hparams["steps_per_epoch"],
        bootstrap_prop=hparams["bootstrap_prop"],
        n_boostrap_heads=hparams["n_heads"],
        verbose=True,
    )

    _eval_callback = eval_callback(
        lambda: gym.make(args.env_name),
        model.select_actions,
        n_envs=25,
        n_eval_episodes=25,
        n_test_episodes=100,
        seed=args.seed,
        mean_return_in_progress_bar=True,
    )

    loggers = create_tb_and_csv_logger(f"D4rl/pesac/{args.env_name}")
    check_point_callback = create_return_based_model_checkpoint(loggers[0].log_dir)

    trainer = pl.Trainer(
        gpus=1 if args.gpu else 0,
        fast_dev_run=args.fast_dev_run,
        max_epochs=max_epochs(hparams["max_steps"], hparams["steps_per_epoch"]),
        callbacks=[_eval_callback, check_point_callback],
        logger=loggers,
        checkpoint_callback=True,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())


def d4rl_gym_datasets():
    datasets = []
    for agent in ["hopper", "halfcheetah", "walker2d"]:
        for dataset in [
            "random",
            "medium",
            "expert",
            "medium-expert",
            "medium-replay",
            "full-replay",
        ]:
            datasets.append(f"{agent}-{dataset}-v2")
    return datasets


def env_defaults(args):
    if (
        "maze2d" in args.env_name
        or "halfcheetah" in args.env_name
        or "walker" in args.env_name
        or "hopper" in args.env_name
    ):
        return {
            "lr": 3e-4,
            "gamma": 0.99,
            "sync_rate": 1,
            "tau": 0.005,
            "grad_norm_max": 10,
            # Uncertainty optimization
            "policy_uncertainty_weight": 1,
            "policy_uncertainty_weight_target": 1,
            "policy_uncertainty_weight_auto_tune": True,
            # Network
            "n_heads": 15,
            "n_hidden_units": (256, 256),
            "auto_alpha_tuning": True,
            "init_alpha": 1,
            "prior_weight": 10,
            # data module
            "batch_size": 256,
            "steps_per_epoch": 5000,
            # boostrapping
            "bootstrap_prop": 0.8,
        }

    raise ValueError(f"No config for {args.env_name}")


if __name__ == "__main__":
    main()

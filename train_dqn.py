import copy
import pprint

import gym
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from src.agents.DQN import DQN
from src.builders import eval_callback, off_policy_dataset
from src.utils.logging import (create_return_based_model_checkpoint,
                               create_tb_and_csv_logger, max_epochs)


def main() -> None:
    from argparse import ArgumentParser

    pp = pprint.PrettyPrinter(indent=2)
    parser = ArgumentParser()

    parser.add_argument("--gpu", default=True, type=bool)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    parser.add_argument(
        "--env_name",
        type=str,
        default="Breakout-MinAtar-chw-v0",
        choices=[
            "Breakout-MinAtar-chw-v0",
            "SpaceInvaders-MinAtar-chw-v0",
            "CartPole-v1",
            "CartPole-v0",
            "LunarLander-v2",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    env = gym.make(args.env_name)

    config = env_defaults(args)
    hparams = {**config, **copy.deepcopy(args.__dict__)}
    pp.pprint(hparams)

    model = DQN(
        observation_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        is_online=True,
        **hparams,
    )

    data_module, callbacks = off_policy_dataset(
        lambda: gym.make(args.env_name),
        model.select_online_actions,
        capacity=hparams["capacity"],
        n_populate_steps=hparams["n_populate_steps"],
        steps_per_epoch=hparams["steps_per_epoch"],
        batch_size=hparams["batch_size"],
    )

    _eval_callback = eval_callback(
        lambda: gym.make(args.env_name),
        model.select_actions,
        n_envs=10,
        n_eval_episodes=10,
        seed=args.seed,
        mean_return_in_progress_bar=True,
    )

    loggers = create_tb_and_csv_logger(f"online/dqn/{args.env_name}")
    check_point_callback = create_return_based_model_checkpoint(loggers[0].log_dir)

    trainer = pl.Trainer(
        gpus=1 if args.gpu else 0,
        fast_dev_run=args.fast_dev_run,
        max_epochs=max_epochs(hparams["max_steps"], hparams["steps_per_epoch"]),
        callbacks=callbacks + [_eval_callback, check_point_callback],
        logger=loggers,
        checkpoint_callback=True,
        gradient_clip_val=10,
    )

    trainer.fit(model, data_module)


def env_defaults(args):
    if "MinAtar" in args.env_name:
        return {
            "lr": 0.00025,
            "gamma": 0.99,
            "sync_rate": 1000,
            "tau": 1,
            # Exploration
            "eps_start": 1.0,
            "eps_frames": 1_000_000,
            "eps_min": 0.1,
            # Network
            "n_hidden_units": (128, 128),
            "filters": (16,),
            "kernel_sizes": (3,),
            "strides": (1,),
            # data module
            "max_steps": 5_000_000,
            "capacity": 1_000_000,
            "n_populate_steps": 10_000,
            "steps_per_epoch": 1000,
            "batch_size": 32,
        }
    if "CartPole" in args.env_name:
        return {
            "lr": 0.00025,
            "gamma": 0.99,
            "sync_rate": 500,
            "tau": 1,
            # Exploration
            "eps_start": 1.0,
            "eps_frames": 20_000,
            "eps_min": 0.1,
            # Network
            "n_hidden_units": (256, 256),
            "filters": None,
            "kernel_sizes": None,
            "strides": None,
            # data module
            "max_steps": 100_000,
            "capacity": 100_000,
            "n_populate_steps": 1000,
            "steps_per_epoch": 1000,
            "batch_size": 32,
        }

    if "LunarLander-v2" == args.env_name:
        return {
            "lr": 0.00025,
            "gamma": 0.99,
            "sync_rate": 1000,
            "tau": 1,
            # Exploration
            "eps_start": 1.0,
            "eps_frames": 100_000,
            "eps_min": 0.1,
            # Network
            "n_hidden_units": (256, 256),
            "filters": None,
            "kernel_sizes": None,
            "strides": None,
            # data module
            "max_steps": 500_000,
            "capacity": 100_000,
            "n_populate_steps": 1000,
            "steps_per_epoch": 1000,
            "batch_size": 32,
        }

    raise ValueError(f"No config for {args.env_name}")


if __name__ == "__main__":
    main()

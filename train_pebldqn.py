import copy
import pprint

import gym
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from src.agents.PEBLDQN import PEBLDQN
from src.builders import eval_callback
from src.datasets.FDPODataModule import FDPODataModule
from src.environments.custom.minatar_envs import register_minatar_envs
from src.utils.logging import (create_return_based_model_checkpoint,
                               create_tb_and_csv_logger, max_epochs)


def main() -> None:
    from argparse import ArgumentParser

    register_minatar_envs()

    pp = pprint.PrettyPrinter(indent=2)
    parser = ArgumentParser()

    parser.add_argument("--gpu", default=True, type=bool)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    parser.add_argument("--eps", default=0.0, type=float)
    parser.add_argument("--dataset_size", default=50_000, type=int)
    parser.add_argument("--max_steps", default=75_000, type=int)
    parser.add_argument(
        "--env_name",
        type=str,
        default="SpaceInvaders-MinAtar-chw-v0",
        choices=["SpaceInvaders-MinAtar-chw-v0", "Breakout-MinAtar-chw-v0"],
    )

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    assert 0.0 <= args.eps <= 1.0

    seed_everything(args.seed)

    env = gym.make(args.env_name)

    config = env_defaults(args)
    hparams = {**config, **copy.deepcopy(args.__dict__)}
    pp.pprint(hparams)

    model = PEBLDQN(
        observation_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        **hparams,
    )

    data_module = FDPODataModule(
        agent="DQN",
        env_name=hparams["env_name"],
        seed=hparams["seed"],
        dataset_size=hparams["dataset_size"],
        eps=hparams["eps"],
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

    eps = str(round(float(hparams["eps"]), 2)).replace(".", "_")
    loggers = create_tb_and_csv_logger(
        f"FDPO/pedqn/{args.env_name}/eps_{eps}/{hparams['dataset_size']}/{args.seed}"
    )
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


def env_defaults(args):
    if "MinAtar" in args.env_name:
        return {
            # Network
            "n_heads": 10,
            "n_hidden_units": (128, 128),
            "filters": (16,),
            "kernel_sizes": (3,),
            "strides": (1,),
            # data module
            "batch_size": 256,
            "dataset_size": 50_000,
            "steps_per_epoch": 5000,
            # boostrapping
            "bootstrap_prop": 0.8,
        }

    raise ValueError(f"No config for {args.env_name}")


if __name__ == "__main__":
    main()

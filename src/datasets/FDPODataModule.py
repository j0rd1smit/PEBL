import os
from pathlib import Path
from typing import Any, Optional

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from src.datasets.samplers.UniformSampler import UniformSampler
from src.environmental.EnvironmentLoop import EnvironmentLoop
from src.storage.UniformReplayBuffer import UniformReplayBuffer
from pytorch_lightning import seed_everything
from src.datasets.boostrapping import Bootstrapping, create_bootstrapping_mask
from src.agents.DQN import DQN
from src.agents.SAC import SAC
from src.utils import relative_to_file
from torch.utils.data import DataLoader


class FDPODataModule(pl.LightningDataModule):
    def __init__(
        self,
        env_name: str,
        agent: str,
        seed: int,
        dataset_size: int,
        eps: float,
        *,
        bootstrap_prop: float = 0.0,
        n_boostrap_heads: int = 0,
        batch_size: int = 256,
        steps_per_epoch: int = 1000,
        pin_memory: bool = True,
        n_workers: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.env_name = env_name
        self.agent = agent
        self.seed = seed
        self.dataset_size = dataset_size
        self.eps = eps

        self.buffer = None
        self.sampler = None

        self.bootstrap_prop = bootstrap_prop
        self.n_boostrap_heads = n_boostrap_heads
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.pin_memory = pin_memory
        self.n_workers = n_workers
        self.verbose = verbose

    @property
    def path_to_data(self) -> str:
        eps = str(round(float(self.eps), 2)).replace(".", "_")
        return str(
            relative_to_file(
                __file__,
                f"../../data/dataset/FDPO/{self.env_name}/eps_{eps}/{self.dataset_size}/{self.seed}/dataset.pt",
            ).resolve()
        )

    def prepare_data(self, *args: Any, **kwargs: Any) -> Any:
        if not os.path.exists(self.path_to_data):
            Path(self.path_to_data).parent.mkdir(parents=True, exist_ok=False)
            print("[INFO] dataset doesn't exists creating new one.")
            self._create_buffer()
            print("[INFO] dataset created and stored:", self.path_to_data)

    def _create_buffer(self):
        root = relative_to_file(__file__, f"../../data/agents/online/{self.agent}/{self.env_name}").resolve()
        checkpoint_path = list(root.rglob("last.ckpt"))[0]

        if "dqn" in self.agent.lower():
            model = DQN.load_from_checkpoint(str(checkpoint_path), hparams_file=str(root / "hparams.yaml"))
        elif "sac" in self.agent.lower():
            model = SAC.load_from_checkpoint(str(checkpoint_path), hparams_file=str(root / "hparams.yaml"))
        else:
            raise Exception("Unknown", self.agent)

        seed_everything(self.seed)

        env = gym.make(model.hparams.env_name)

        def policy(x):
            if np.random.random() <= self.eps:
                return torch.randint(low=0, high=env.action_space.n, size=x.shape[:1])
            return model.select_actions(x)

        buffer = UniformReplayBuffer(capacity=self.dataset_size)
        env_loop = EnvironmentLoop(env, policy)
        env_loop.seed(self.seed)

        _it_wrapper = tqdm.tqdm if self.verbose else lambda x: x

        for _ in _it_wrapper(range(self.dataset_size)):
            batch = env_loop.step()
            buffer.append(batch)

        buffer.save(self.path_to_data)

    def setup(self, stage: Optional[str] = None) -> Any:
        self.buffer = UniformReplayBuffer.load(self.path_to_data)
        if self.bootstrap_prop > 0.0:
            assert self.n_boostrap_heads > 0
            mask = create_bootstrapping_mask(len(self.buffer), self.n_boostrap_heads, self.bootstrap_prop)
            self.buffer.buffer[Bootstrapping.MASK] = mask

        samples_per_epoch = self.steps_per_epoch * self.batch_size
        self.sampler = UniformSampler(self.buffer, samples_per_epoch)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.buffer,
            sampler=self.sampler,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        dataset = torch.utils.data.TensorDataset(torch.zeros([1]))
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

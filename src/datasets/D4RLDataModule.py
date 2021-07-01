from typing import Any, Optional

import d4rl
import gym
import pytorch_lightning as pl
import torch
from src.datasets.boostrapping import Bootstrapping, create_bootstrapping_mask
from torch.utils.data import DataLoader

from src.datasets.samplers.UniformSampler import UniformSampler
from src.environmental.SampleBatch import SampleBatch
from src.storage.UniformReplayBuffer import UniformReplayBuffer


class D4RLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        env_name: str,
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

        self.buffer = None
        self.sampler = None

        self.bootstrap_prop = bootstrap_prop
        self.n_boostrap_heads = n_boostrap_heads
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.pin_memory = pin_memory
        self.n_workers = n_workers
        self.verbose = verbose

    def setup(self, stage: Optional[str] = None) -> Any:
        env = gym.make(self.env_name)
        dataset = d4rl.qlearning_dataset(env)

        data = {
            SampleBatch.OBSERVATIONS: torch.from_numpy(dataset["observations"]),
            SampleBatch.ACTIONS: torch.from_numpy(dataset["actions"]),
            SampleBatch.REWARDS: torch.from_numpy(dataset["rewards"]),
            SampleBatch.DONES: torch.from_numpy(dataset["terminals"]).float(),
            SampleBatch.OBSERVATION_NEXTS: torch.from_numpy(dataset["next_observations"]),
        }
        del dataset
        del env

        capacity = len(data[SampleBatch.OBSERVATIONS])

        if self.bootstrap_prop > 0.0:
            assert self.n_boostrap_heads > 0
            mask = create_bootstrapping_mask(capacity, self.n_boostrap_heads, self.bootstrap_prop)
            data[Bootstrapping.MASK] = mask

        capacity = len(data[SampleBatch.OBSERVATIONS])
        self.buffer = UniformReplayBuffer(capacity=capacity, size=capacity, buffer=data)

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

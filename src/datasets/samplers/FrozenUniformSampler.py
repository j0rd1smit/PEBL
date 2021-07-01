import random
from typing import Iterator

from torch.utils.data import Sampler


class FrozenUniformSampler(Sampler):
    def __init__(
        self,
        buffer_length: int,
        samples_per_epoch: int,
    ) -> None:
        super().__init__(None)
        self.samples_per_epoch = samples_per_epoch
        self.idxs = list(range(buffer_length))
        random.shuffle(self.idxs)
        self.pointer = 0

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.samples_per_epoch):
            yield int(self.idxs[self.pointer])
            self.pointer = (self.pointer + 1) % len(self.idxs)

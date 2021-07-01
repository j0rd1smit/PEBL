from typing import Iterator

from torch.utils.data import Sampler

from src.storage.IBuffer import Buffer


class EntireBufferSampler(Sampler):
    def __init__(
        self,
        buffer: Buffer,
        samples_per_epoch: int,
    ) -> None:
        super().__init__(None)
        self.buffer = buffer
        self.samples_per_epoch = samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        idx = 0
        for _ in range(self.samples_per_epoch):
            yield idx
            idx = (idx + 1) % max(1, len(self.buffer))

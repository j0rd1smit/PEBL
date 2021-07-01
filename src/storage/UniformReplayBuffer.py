from typing import Dict, Optional

import torch

from src.environmental.SampleBatch import SampleBatch
from src.storage.IBuffer import DynamicBuffer


class UniformReplayBuffer(DynamicBuffer):
    EXCLUDED_KEYS = [SampleBatch.IDX]

    def __init__(
        self,
        capacity: int,
        buffer: Optional[Dict[str, torch.Tensor]] = None,
        pointer: Optional[int] = None,
        size: Optional[int] = None,
    ) -> None:
        assert (size is None and buffer is None) or (size is not None and buffer is not None)

        self._capacity = capacity

        if buffer is None:
            self.buffer: Dict[str, torch.Tensor] = {}
            self.size = 0
        else:
            for k, v in buffer.items():
                assert len(v) <= capacity, f"datasize is large than capacity for {k}"
            self.buffer: Dict[str, torch.Tensor] = buffer
            self.size = size

        if pointer is None:
            self.pointer = min(self.size, self.capacity)
        else:
            assert pointer <= self.size
            self.pointer = pointer

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, batch: SampleBatch) -> None:
        assert len(batch[SampleBatch.REWARDS].shape) > 0, "Assumes that input indexable, please batch the results"

        if len(self.buffer) == 0:
            for k, v in batch.items():
                if k not in self.EXCLUDED_KEYS:
                    shape = (self.capacity,) + v.shape[1:]
                    self.buffer[k] = torch.zeros(shape, dtype=v.dtype)
        for i in range(batch.n_samples):
            for k, v in batch.items():
                if k not in self.EXCLUDED_KEYS:
                    self.buffer[k][self.pointer] = v[i]

            self.pointer = (self.pointer + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def __getitem__(self, idxs: int) -> SampleBatch:
        assert isinstance(idxs, int)
        assert idxs < len(self), f"{idxs} > {len(self)}"
        batch = {k: self.buffer[k][idxs] for k in self.buffer}
        batch[SampleBatch.IDX] = torch.tensor(idxs)

        return SampleBatch(batch)

    def clear(self) -> None:
        self.pointer = 0
        self.size = 0

    def save(
        self,
        path: str,
        include_meta_data: bool = True,
    ) -> None:
        data = {
            "buffer": self.buffer,
        }

        if include_meta_data:
            data["pointer"] = self.pointer
            data["capacity"] = self.capacity
            data["size"] = self.size

        torch.save(data, path)

    @staticmethod
    def load(path: str):
        data = torch.load(path)
        pointer = data.get("pointer", None)
        capacity = data.get("capacity", None)
        buffer = data["buffer"]
        size = data.get("size", None)

        return UniformReplayBuffer(
            capacity=capacity,
            pointer=pointer,
            buffer=buffer,
            size=size,
        )

    def __len__(self) -> int:
        return self.size

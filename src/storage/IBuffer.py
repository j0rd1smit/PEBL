import abc

from torch.utils.data import Dataset

from src.environmental.SampleBatch import SampleBatch


class Buffer(abc.ABC, Dataset):
    @abc.abstractmethod
    def __getitem__(self, idxs: int) -> SampleBatch:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class DynamicBuffer(Buffer):
    @abc.abstractmethod
    def append(self, batch: SampleBatch) -> None:
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        pass

    @property
    @abc.abstractmethod
    def capacity(self) -> int:
        pass

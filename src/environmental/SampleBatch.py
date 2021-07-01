from typing import Dict, List

import torch


class SampleBatch(dict):
    OBSERVATIONS = "OBSERVATIONS"
    ACTIONS = "ACTIONS"
    REWARDS = "REWARDS"
    DONES = "DONES"
    OBSERVATION_NEXTS = "OBSERVATION_NEXTS"
    IDX = "IDX"
    VALE_PREDICTIONS = "VALE_PREDICTIONS"
    EPS_ID = "EPS_ID"

    def __init__(self, _dict: Dict[str, torch.Tensor]) -> None:
        assert len(_dict) > 0
        for k, v in _dict.items():
            assert isinstance(v, torch.Tensor), "A sample batch should only contain tensors."

        super().__init__(_dict)

    @property
    def n_samples(self) -> int:
        lenght = -1
        for k, v in self.items():
            if lenght == -1:
                lenght = len(v)

            assert len(v) == lenght, f"k={k} len(v) = {len(v)} lenght={lenght}"
        assert lenght != -1, "Sample batch cannot be empty"

        return lenght

    @staticmethod
    def concat_samples(samples: List["SampleBatch"]) -> "SampleBatch":
        concat_samples = []
        for s in samples:
            concat_samples.append(s)

        out = {}
        for k in concat_samples[0].keys():
            out[k] = torch.cat([s[k] for s in concat_samples], 0)

        return SampleBatch(out)

    def split_by_episode(self) -> List["SampleBatch"]:
        assert SampleBatch.EPS_ID in self

        episode_ids = torch.unique(self[SampleBatch.EPS_ID])
        slices = []
        for episode_id in episode_ids:
            mask = self[SampleBatch.EPS_ID] == episode_id
            slices.append(SampleBatch({k: v[mask] for k, v in self.items()}))

        return slices

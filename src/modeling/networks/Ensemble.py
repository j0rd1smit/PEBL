from typing import List

import torch


class Ensemble(torch.nn.Module):
    def __init__(
        self,
        models: List[torch.nn.Module],
    ) -> None:
        super().__init__()

        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [model(x) for model in self.models]
        return torch.stack(x, dim=1)

    def __getitem__(self, item: int) -> torch.nn.Module:
        return self.models[item]

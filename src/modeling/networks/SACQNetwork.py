from typing import Optional

import pytorch_lightning as pl
import torch


class SACQNetwork(pl.LightningModule):
    def __init__(
        self,
        *,
        fc: torch.nn.Module,
        encoder: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.fc = fc

    def forward(self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = observations
        if self.encoder is not None:
            x = self.encoder(x)

        if actions is not None:
            x = torch.cat([x, actions], -1)

        return self.fc(x)

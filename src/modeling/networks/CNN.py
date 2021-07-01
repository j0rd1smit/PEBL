from typing import Any, List, Optional, Sequence, Union

import torch


class CNN(torch.nn.Module):
    def __init__(
        self,
        n_input_filters: int,
        filters: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        flatten: bool = True,
        dropout_probs: Optional[Union[float, Sequence[float]]] = None,
        activation: Any = None,
    ) -> None:
        super().__init__()

        if isinstance(dropout_probs, float) or isinstance(dropout_probs, int):
            dropout_probs = [dropout_probs for _ in filters]
        if dropout_probs is None:
            dropout_probs = [0 for _ in filters]

        filters = [n_input_filters] + list(filters)
        sequence: List[torch.nn.Module] = []

        for i in range(len(filters) - 1):
            conv2d = torch.nn.Conv2d(
                filters[i],
                filters[i + 1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
            )
            sequence.append(conv2d)
            if activation is None:
                sequence.append(torch.nn.ReLU())
            else:
                sequence.append(activation())

            if dropout_probs[i] > 0:
                sequence.append(torch.nn.Dropout2d(p=dropout_probs[i]))

        if flatten:
            sequence.append(torch.nn.Flatten())

        self.model = torch.nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

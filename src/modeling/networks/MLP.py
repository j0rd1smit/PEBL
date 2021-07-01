from typing import Any, List, Optional, Sequence, Union

import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        *,
        n_inputs: int,
        n_hidden_units: Optional[Union[int, Sequence[int]]] = None,
        n_outputs: Optional[int] = None,
        dropout_probs: Optional[Union[float, Sequence[float]]] = None,
        activation: Any = None,
    ) -> None:
        super().__init__()

        if isinstance(n_hidden_units, int):
            n_hidden_units = [n_hidden_units]

        if n_hidden_units is None:
            n_hidden_units = []
        n_hidden_units = list(n_hidden_units)

        if isinstance(dropout_probs, float) or isinstance(dropout_probs, int):
            dropout_probs = [dropout_probs for _ in n_hidden_units]
        if dropout_probs is None:
            dropout_probs = [0 for _ in n_hidden_units]

        dropout_probs = list(dropout_probs)

        assert len(dropout_probs) == len(
            n_hidden_units
        ), f"Dropout probs must have same length as hidden_sizes but {len(dropout_probs)} != {len(n_hidden_units)}"
        assert all(
            (0 <= dropout_prob < 1 for dropout_prob in dropout_probs)
        ), f"Dropout prob must be in range [0, 1] but probs are {dropout_probs}"

        sequence: List[torch.nn.Module] = []
        for i, (n_in, n_out, dropout_prob) in enumerate(zip([n_inputs] + list(n_hidden_units[:-1]), n_hidden_units, dropout_probs)):
            layer = []
            layer.append(self._linear_layer(n_in, n_out))
            if activation is None:
                layer.append(torch.nn.ELU())
            else:
                layer.append(activation())

            if dropout_prob > 0:
                layer.append(torch.nn.Dropout(p=dropout_prob))

            sequence.append(torch.nn.Sequential(*layer))

        if n_outputs is not None:
            last_size = n_hidden_units[-1] if n_hidden_units else n_inputs
            sequence.append(self._linear_layer(last_size, n_outputs))

        self.model = torch.nn.Sequential(*sequence)

        self._output_size = n_hidden_units[-1] if n_outputs is None else n_outputs

    def _linear_layer(self, n_in: int, n_out: int) -> torch.nn.Module:
        return torch.nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

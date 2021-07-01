from typing import Callable, Dict, Optional, Union

import numpy as np
import torch

Seed = Optional[int]

Action = torch.Tensor
Observation = Union[np.ndarray]


Policy = Callable[[torch.Tensor], torch.Tensor]
FetchAgentInfo = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]

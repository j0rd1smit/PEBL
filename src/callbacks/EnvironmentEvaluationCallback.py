import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from pytorch_lightning import Callback, LightningModule

from src.environmental.EnvironmentLoop import EnvironmentLoop
from src.environmental.SampleBatch import SampleBatch

ScoreMapper = Callable[[np.ndarray], np.ndarray]
LengthMapper = Callable[[np.ndarray], np.ndarray]

default_return_mappers = {
    "return/mean": np.mean,
    "return/min": np.min,
    "return/max": np.max,
    "return/median": np.median,
    "return/std": np.std,
}

default_length_mappers = {
    "length/mean": np.mean,
    "length/min": np.min,
    "length/max": np.max,
    "length/std": np.std,
}


class EnvironmentEvaluationCallback(Callback):
    def __init__(
        self,
        env_loop: EnvironmentLoop,
        *,
        n_eval_episodes: int = 10,
        n_test_episodes: int = 250,
        return_mappers: Optional[Dict[str, ScoreMapper]] = None,
        length_mappers: Optional[Dict[str, LengthMapper]] = None,
        seed: Optional[int] = None,
        to_eval: bool = False,
        logging_prefix: str = "Evaluation",
        mean_return_in_progress_bar: bool = True,
    ) -> None:
        self.env_loop = env_loop
        self.n_eval_episodes = n_eval_episodes
        self.n_test_episodes = n_test_episodes
        self.return_mappers = (
            return_mappers
            if return_mappers is not None
            else cast(Dict[str, ScoreMapper], copy.deepcopy(default_return_mappers))
        )
        self.length_mappers = (
            length_mappers
            if length_mappers is not None
            else cast(Dict[str, LengthMapper], copy.deepcopy(default_length_mappers))
        )
        self.seed = seed
        self.to_eval = to_eval
        self.logging_prefix = logging_prefix
        self.mean_return_in_progress_bar = mean_return_in_progress_bar

    def on_train_epoch_end(self, trainer, pl_module: LightningModule, outputs: Any) -> None:
        self.env_loop.seed(self.seed)
        was_in_training_mode = pl_module.training
        if self.to_eval:
            pl_module.eval()

        returns: List[float] = []
        lengths: List[float] = []

        while len(returns) < self.n_eval_episodes:
            self.env_loop.reset()
            _lengths, _returns = self._eval_env_run()
            returns = returns + _returns
            lengths = lengths + _lengths

        returns_arr = np.array(returns)
        lengths_arr = np.array(lengths)

        if self.to_eval and was_in_training_mode:
            pl_module.train()

        for k, mapper in self.return_mappers.items():
            v: Any = mapper(returns_arr)
            pl_module.log(self.logging_prefix + "/" + k, v, prog_bar=False)

        for k, mapper in self.length_mappers.items():
            v: Any = mapper(lengths_arr)  # type: ignore
            pl_module.log(self.logging_prefix + "/" + k, v, prog_bar=False)

        if self.mean_return_in_progress_bar:
            pl_module.log("return", np.mean(returns), prog_bar=True)

    def _eval_env_run(self) -> Tuple[List[float], List[float]]:
        dones = [False for _ in range(self.env_loop.n_enviroments)]
        returns = [0.0 for _ in range(self.env_loop.n_enviroments)]
        lengths = [0 for _ in range(self.env_loop.n_enviroments)]

        while not all(dones):
            batch = self.env_loop.step()

            for i in range(self.env_loop.n_enviroments):
                if dones[i]:
                    continue
                dones[i] = bool(batch[SampleBatch.DONES][i])
                returns[i] = returns[i] + float(batch[SampleBatch.REWARDS][i])
                lengths[i] += 1

        return list(lengths), list(returns)

    def on_test_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        if self.n_test_episodes <= 0:
            return

        self.env_loop.seed(self.seed)
        was_in_training_mode = pl_module.training
        if self.to_eval:
            pl_module.eval()

        returns: List[float] = []
        lengths: List[float] = []

        while len(returns) < self.n_test_episodes:
            self.env_loop.reset()
            _lengths, _returns = self._eval_env_run()
            returns = returns + _returns
            lengths = lengths + _lengths

        returns_arr = np.array(returns)
        lengths_arr = np.array(lengths)

        if self.to_eval and was_in_training_mode:
            pl_module.train()

        for k, mapper in self.return_mappers.items():
            v: Any = mapper(returns_arr)
            pl_module.log(self.logging_prefix + "/test/" + k, v, prog_bar=False)

        for k, mapper in self.length_mappers.items():
            v: Any = mapper(lengths_arr)  # type: ignore
            pl_module.log(self.logging_prefix + "/test/" + k, v, prog_bar=False)

import math
from typing import Any, Callable, Optional

from pytorch_lightning import Callback, LightningModule

from src.environmental.EnvironmentLoop import EnvironmentLoop
from src.environmental.SampleBatch import SampleBatch
from src.storage.IBuffer import DynamicBuffer

PostProcessFunction = Callable[[SampleBatch], SampleBatch]

# TODO propably best to move this to sampler??
class OnlineDataCollectionCallback(Callback):
    def __init__(
        self,
        buffer: DynamicBuffer,
        env_loop: EnvironmentLoop,
        n_samples_per_step: int,
        n_populate_steps: int,
        post_process_function: Optional[PostProcessFunction] = None,
        *,
        clear_buffer_before_gather: bool = False,
    ) -> None:
        self.buffer = buffer
        self.env_loop = env_loop

        self.n_samples_per_step = n_samples_per_step
        self.n_populate_steps = n_populate_steps
        self.post_process_function = post_process_function
        self.clear_buffer_before_gather = clear_buffer_before_gather

    def on_fit_start(self, trainer: Any, pl_module: LightningModule) -> None:
        assert (
            self.n_populate_steps <= self.buffer.capacity
        ), f"Current config is not possible. n_populate_steps must be <= buffer.capacity but {self.n_populate_steps} > {self.buffer.capacity}"
        while len(self.buffer) < self.n_populate_steps:
            self._add_batch_to_buffer(sample=True)

    def _add_batch_to_buffer(self, *, sample: bool = False) -> None:
        batch = SampleBatch.concat_samples(
            [
                self.env_loop.step() if not sample else self.env_loop.sample()
                for _ in range(math.ceil(self.n_samples_per_step / self.env_loop.n_enviroments))
            ]
        )
        batches_per_episode = batch.split_by_episode()

        for batch_per_episode in batches_per_episode:
            if self.post_process_function is not None:
                batch_per_episode = self.post_process_function(batch_per_episode)

            self.buffer.append(batch_per_episode)

    def on_epoch_start(self, trainer: Any, pl_module: LightningModule) -> None:
        if len(self.buffer) == 0:
            self._add_batch_to_buffer()

    def on_batch_end(self, trainer: Any, pl_module: LightningModule) -> None:
        if self.clear_buffer_before_gather:
            self.buffer.clear()
        self._add_batch_to_buffer()

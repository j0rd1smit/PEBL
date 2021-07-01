from typing import Callable, List, Optional, Tuple

import gym
from gym.vector import SyncVectorEnv
from pytorch_lightning import Callback

from src.callbacks.EnvironmentEvaluationCallback import EnvironmentEvaluationCallback
from src.callbacks.OnlineDataCollectionCallback import OnlineDataCollectionCallback, PostProcessFunction
from src.datasets.OnlineDataModule import OnlineDataModule
from src.datasets.samplers.EntireBufferSampler import EntireBufferSampler
from src.datasets.samplers.UniformSampler import UniformSampler
from src.environmental.EnvironmentLoop import EnvironmentLoop
from src.storage.UniformReplayBuffer import UniformReplayBuffer
from src.types import FetchAgentInfo, Policy

EnvBuilder = Callable[[], gym.Env]


def on_policy_dataset(
    env_builder: EnvBuilder,
    select_online_actions: Policy,
    fetch_agent_info: Optional[FetchAgentInfo] = None,
    # batch
    batch_size: int = 4000,
    # online callback
    n_envs: int = 10,
    steps_per_epoch: int = 5000,
    # post processing
    post_process_function: Optional[PostProcessFunction] = None,
) -> Tuple[OnlineDataModule, List[Callback]]:
    buffer = UniformReplayBuffer(batch_size)

    samples_per_epoch = steps_per_epoch * batch_size
    sampler = EntireBufferSampler(buffer, samples_per_epoch)

    data_module = OnlineDataModule(buffer, batch_size, sampler=sampler, pin_memory=True, n_workers=0)

    online_env = _build_env(env_builder, n_envs)

    n_samples_per_step = batch_size
    env_loop = EnvironmentLoop(online_env, select_online_actions, fetch_agent_info=fetch_agent_info)

    online_step_callback = OnlineDataCollectionCallback(
        buffer,
        env_loop,
        n_samples_per_step=n_samples_per_step,
        n_populate_steps=0,
        post_process_function=post_process_function,
        clear_buffer_before_gather=True,
    )

    return data_module, [online_step_callback]


def _build_env(env_builder: EnvBuilder, n_envs: int) -> gym.Env:
    if n_envs > 1:
        return SyncVectorEnv([env_builder for _ in range(n_envs)])
    else:
        return env_builder()


def off_policy_dataset(
    env_builder: EnvBuilder,
    select_online_actions: Policy,
    fetch_agent_info: Optional[FetchAgentInfo] = None,
    # buffer
    capacity: int = 100_000,
    # batch
    batch_size: int = 32,
    # online callback
    n_envs: int = 1,
    steps_per_epoch: int = 5000,
    n_populate_steps: int = 10000,
    # post processing
    post_process_function: Optional[PostProcessFunction] = None,
) -> Tuple[OnlineDataModule, List[Callback]]:
    buffer = UniformReplayBuffer(capacity)

    samples_per_epoch = steps_per_epoch * batch_size
    sampler = UniformSampler(buffer, samples_per_epoch)

    data_module = OnlineDataModule(buffer, batch_size, sampler=sampler, pin_memory=True, n_workers=0)

    online_env = _build_env(env_builder, n_envs)

    n_samples_per_step = batch_size
    env_loop = EnvironmentLoop(online_env, select_online_actions, fetch_agent_info=fetch_agent_info)

    online_step_callback = OnlineDataCollectionCallback(
        buffer,
        env_loop,
        n_samples_per_step=n_samples_per_step,
        n_populate_steps=n_populate_steps,
        post_process_function=post_process_function,
        clear_buffer_before_gather=False,
    )

    return data_module, [online_step_callback]


def eval_callback(
    env_builder: EnvBuilder,
    select_actions: Policy,
    seed: Optional[int] = None,
    n_envs: int = 1,
    n_eval_episodes: int = 10,
    n_test_episodes: int = 100,
    to_eval: bool = False,
    logging_prefix: str = "Evaluation",
    mean_return_in_progress_bar: bool = True,
) -> EnvironmentEvaluationCallback:
    env = _build_env(env_builder, n_envs)

    env_loop = EnvironmentLoop(env, select_actions)
    return EnvironmentEvaluationCallback(
        env_loop,
        n_eval_episodes=n_eval_episodes,
        n_test_episodes=n_test_episodes,
        to_eval=to_eval,
        seed=seed,
        logging_prefix=logging_prefix,
        mean_return_in_progress_bar=mean_return_in_progress_bar,
    )

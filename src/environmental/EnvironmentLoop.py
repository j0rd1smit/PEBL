from typing import Callable, Optional, Union

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple
from gym.vector import AsyncVectorEnv, VectorEnv

from src.environmental.SampleBatch import SampleBatch
from src.types import FetchAgentInfo, Observation, Policy, Seed


class EnvironmentLoop:
    def __init__(self, env: gym.Env, policy: Policy, fetch_agent_info: Optional[FetchAgentInfo] = None) -> None:
        self.env = env
        self.policy = policy
        self.fetch_agent_info = fetch_agent_info

        self._obs = np.array(self.env.reset())

        self._is_vectorized = isinstance(
            env,
            VectorEnv,
        )

        self._done = not self._is_vectorized
        self._episode_ids = np.arange(self.n_enviroments, dtype=np.int64)

        if self._is_vectorized:
            self._expected_observation_shape = (self.n_enviroments,) + self.env.envs[0].observation_space.shape
        else:
            self._expected_observation_shape = (1,) + self.env.observation_space.shape
            self._obs = np.expand_dims(self._obs, 0)

        if isinstance(env.action_space, Discrete):
            self._is_discrete_action_space = True
            self._expected_action_shape = (1,) + self.env.action_space.shape
        elif isinstance(env.action_space, Box):
            self._is_discrete_action_space = False
            self._expected_action_shape = (1,) + self.env.action_space.shape
        elif isinstance(env.action_space, Tuple):
            self._is_discrete_action_space = isinstance(
                env.envs[0].action_space,
                Discrete,
            )
            self._expected_action_shape = (self.n_enviroments,) + self.env.envs[0].action_space.shape
        else:
            raise Exception("Unknown action space", env.action_space)

        assert not isinstance(env, AsyncVectorEnv), "Async is not supported."

    @property
    def n_enviroments(self) -> int:
        if not self._is_vectorized:
            return 1
        return self.env.observation_space.shape[0]

    def seed(self, seed: Seed = None) -> None:
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.reset()

    def reset(self) -> None:
        self._obs = self.env.reset()
        if not self._is_vectorized:
            self._obs = np.expand_dims(self._obs, 0)
        self._done = False

    def step(self) -> SampleBatch:
        return self._step(self._policy)

    def _policy(self, obs: Observation) -> torch.Tensor:
        obs_tensor = self._cast_obs(obs)

        with torch.no_grad():
            action = self.policy(obs_tensor)
            action = action.cpu()

            return action

    def _step(self, policy: Callable[[Observation], torch.Tensor]) -> SampleBatch:
        if not self._is_vectorized and self._done:
            self.reset()

        obs = np.array(self._obs)
        assert obs.shape == self._expected_observation_shape, f"{ obs.shape} != {self._expected_observation_shape}"

        action = policy(obs)
        _action = action.numpy() if isinstance(action, torch.Tensor) else action
        assert self._expected_action_shape == _action.shape, f"{self._expected_action_shape} != {_action.shape}"

        if not self._is_vectorized:
            obs_next, r, d, _ = self.env.step(_action[0])
        elif self._is_discrete_action_space:
            obs_next, r, d, _ = self.env.step(list(map(int, _action)))
        else:
            obs_next, r, d, _ = self.env.step(_action)

        if not self._is_vectorized:
            obs_next = np.expand_dims(obs_next, 0)
            assert obs_next.shape == self._expected_observation_shape

        batch = {
            SampleBatch.OBSERVATIONS: self._cast_obs(obs),
            SampleBatch.ACTIONS: action,
            SampleBatch.REWARDS: self._batch_if_needed(torch.tensor(r, dtype=torch.float32)),
            SampleBatch.DONES: self._batch_if_needed(torch.tensor(d, dtype=torch.float32)),
            SampleBatch.OBSERVATION_NEXTS: self._cast_obs(obs_next),
            SampleBatch.EPS_ID: torch.from_numpy(self._episode_ids.copy()),
        }

        if self.fetch_agent_info is not None:
            with torch.no_grad():
                agent_info = self.fetch_agent_info(batch)

            for k, v in agent_info.items():
                assert isinstance(v, torch.Tensor)
                assert k not in batch
                batch[k] = v

        if not self._is_vectorized:
            self._done = d
        if np.any(d):
            self._update_episodes_ids_if_needed(d)

        self._obs = obs_next

        return SampleBatch(batch)

    def _cast_obs(self, obs: Observation) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            if obs.dtype in [np.float64]:
                obs = obs.astype(np.float32)

            return torch.from_numpy(obs)

        return torch.tensor(obs)

    def _batch_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_vectorized:
            x = torch.unsqueeze(x, 0)

        return x

    def _update_episodes_ids_if_needed(self, d: Union[bool, np.ndarray]) -> None:
        if isinstance(d, bool):
            d = np.array([d])
        for i in range(self.n_enviroments):
            if d[i]:
                self._episode_ids[i] = self._episode_ids.max() + 1

    def sample(self) -> SampleBatch:
        return self._step(self._sample_policy)

    def _sample_policy(self, _: Observation) -> torch.Tensor:
        if not self._is_vectorized:
            return torch.from_numpy(np.expand_dims(np.array(self.env.action_space.sample()), 0))

        return torch.tensor(self.env.action_space.sample())

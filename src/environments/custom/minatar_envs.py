import gym
import numpy as np
from gym import ObservationWrapper, register
from gym_minatar.envs import AsterixEnv, BreakoutEnv, FreewayEnv, SeaquestEnv, Space_invadersEnv


class MinAtarWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space.shape = (self.observation_space.shape[-1],) + self.observation_space.shape[:-1]

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.transpose(2, 0, 1)


def register_minatar_envs():
    for game in ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]:
        register(id=f"{game}-MinAtar-chw-v0", entry_point=f"src.environments.custom.minatar_envs:CHW{game}Env")


def CHWAsterixEnv() -> gym.Env:
    return MinAtarWrapper(AsterixEnv())


def CHWBreakoutEnv() -> gym.Env:
    return MinAtarWrapper(BreakoutEnv())


def CHWFreewayEnv() -> gym.Env:
    return MinAtarWrapper(FreewayEnv())


def CHWSeaquestEnv() -> gym.Env:
    return MinAtarWrapper(SeaquestEnv())


def CHWSpaceInvadersEnv() -> gym.Env:
    return MinAtarWrapper(Space_invadersEnv())

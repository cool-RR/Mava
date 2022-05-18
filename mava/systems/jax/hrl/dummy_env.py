from typing import List

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from dm_env._environment import TimeStep

from mava.types import OLT
from mava.utils.wrapper_utils import (
    convert_dm_compatible_observations,
    convert_np_type,
    parameterized_restart,
)
from mava.wrappers.env_wrappers import ParallelEnvWrapper


class DummyEnv(ParallelEnvWrapper):
    def __init__(self, num_agents=2, hl_obs_size=(5,), ll_obs_size=(3,)):
        super(DummyEnv, self).__init__()

        self.num_agents = num_agents
        self.hl_obs_size = hl_obs_size
        self.ll_obs_size = ll_obs_size

        self.t = 0

        self.hl_action_shape = (2,)
        self.ll_action_shape = (4,)

    def gen_hl_obs(self):
        return np.random.uniform(size=self.hl_obs_size)

    def gen_ll_obs(self):
        return np.random.uniform(size=self.ll_obs_size)

    def reset(self) -> TimeStep:
        self.t = 0
        return parameterized_restart(
            {agent: 0.0 for agent in self.possible_agents},
            {agent: 1.0 for agent in self.possible_agents},
            {agent: self.gen_hl_obs() for agent in self.possible_agents},
        )

    def step(self, action) -> TimeStep:
        self.t += 1

        step_type = dm_env.StepType.MID if self.t < 50 else dm_env.StepType.LAST
        discount = 1.0 if self.t < 50 else 0.0

        return TimeStep(
            step_type,
            {agent: 1.0 for agent in self.possible_agents},
            {agent: discount for agent in self.possible_agents},
            {agent: self.gen_hl_obs() for agent in self.possible_agents},
        )

    def observation_spec(self):
        return {
            "hl": self.gen_observation_spec(self.gen_hl_obs(), self.hl_action_shape),
            "ll": self.gen_observation_spec(self.gen_ll_obs(), self.ll_action_shape),
        }

    def gen_observation_spec(self, obs, action_shape):
        return {
            agent: OLT(
                observation=specs.BoundedArray(obs.shape, obs[0].dtype, 0.0, 1.0),
                legal_actions=specs.BoundedArray(action_shape, float, -10.0, 10.0),
                terminal=specs.Array((1,), np.float32),
            )
            for agent in self.possible_agents
        }

    def action_spec(self):
        return {
            "hl": {
                agent: specs.DiscreteArray(num_values=self.hl_action_shape[0])
                for agent in self.possible_agents
            },
            "ll": {
                agent: specs.DiscreteArray(num_values=self.ll_action_shape[0])
                for agent in self.possible_agents
            },
        }

    def reward_spec(self):
        return {agent: specs.Array((), np.float32) for agent in self.possible_agents}

    def discount_spec(self):
        return {
            agent: specs.BoundedArray((), np.float32, minimum=0, maximum=1.0)
            for agent in self.possible_agents
        }

    def extra_spec(self):
        return {}

    @property
    def possible_agents(self) -> List:
        return [f"agent_{i}" for i in range(self.num_agents)]

    @property
    def agents(self) -> List:
        return [f"agent_{i}" for i in range(self.num_agents)]

    def env_done(self) -> bool:
        return self.t < 50

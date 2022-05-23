from typing import List

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from dm_env._environment import TimeStep

from mava.systems.jax.hrl.hrl_wrapper import HrlEnvironmentWrapper
from mava.types import OLT
from mava.utils.wrapper_utils import parameterized_restart


class DummyEnv(HrlEnvironmentWrapper):
    def __init__(self, num_agents=2, hl_obs_size=(3,), ll_obs_size=(4,)):
        super(DummyEnv, self).__init__()

        self.num_agents = num_agents
        self.hl_obs_size = hl_obs_size
        self.ll_obs_size = ll_obs_size

        self.t = 0

        self.hl_num_actions = 5
        self.ll_num_actions = 4

    def gen_hl_obs(self):
        return OLT(
            np.random.uniform(size=self.hl_obs_size),
            np.array([True] * self.hl_num_actions),
            np.array([self.env_done()]),
        )

    def gen_ll_obs(self):
        return OLT(
            np.random.uniform(size=self.ll_obs_size),
            np.array([True] * self.ll_num_actions),
            np.array([self.env_done()]),
        )

    def reset(self) -> TimeStep:
        self.t = 0
        return parameterized_restart(
            {agent: 0.0 for agent in self.possible_agents},
            {agent: 1.0 for agent in self.possible_agents},
            {agent: self.gen_hl_obs() for agent in self.possible_agents},
        )

    # TODO (sasha) invert obs level/agent nesting
    def step(self, action) -> TimeStep:
        self.t += 1

        step_type = dm_env.StepType.MID if not self.env_done() else dm_env.StepType.LAST
        discount = 1.0 if not self.env_done() else 0.0

        return TimeStep(
            step_type,
            {agent: 1.0 for agent in self.possible_agents},
            {agent: discount for agent in self.possible_agents},
            {agent: self.gen_hl_obs() for agent in self.possible_agents},
        )

    def ll_timestep(self, ts: TimeStep, hl_actions: np.ndarray) -> TimeStep:
        observations = {}
        rewards = {}

        for agent in ts.observation.keys():
            hl_action = hl_actions[agent]
            if np.ndim(hl_action) == 0:
                hl_action = np.expand_dims(hl_action, 0)

            obs = np.concatenate([ts.observation[agent].observation, hl_action])
            legals = np.array([True] * 4)
            terminal = np.array([self.env_done()])

            observations[agent] = OLT(obs, legals, terminal)
            rewards[agent] = ts.reward[agent] - 0.1

        return TimeStep(ts.step_type, rewards, ts.discount, observations)

    def observation_spec(self):
        return {
            "hl": self.gen_observation_spec(self.gen_hl_obs()),
            "ll": self.gen_observation_spec(self.gen_ll_obs()),
        }

    def gen_observation_spec(self, obs):
        return {
            agent: OLT(
                observation=specs.BoundedArray(
                    obs.observation.shape, obs.observation[0].dtype, 0.0, 1.0
                ),
                legal_actions=specs.BoundedArray(
                    obs.legal_actions.shape, obs.legal_actions.dtype, -10.0, 10.0
                ),
                terminal=specs.Array(obs.terminal.shape, obs.terminal.dtype),
            )
            for agent in self.possible_agents
        }

    def action_spec(self):
        return {
            "hl": {
                agent: specs.DiscreteArray(
                    num_values=self.hl_num_actions, dtype=np.int64
                )
                for agent in self.possible_agents
            },
            "ll": {
                agent: specs.DiscreteArray(
                    num_values=self.ll_num_actions, dtype=np.int64
                )
                for agent in self.possible_agents
            },
        }

    def reward_spec(self):
        return {
            key: {agent: specs.Array((), np.float64) for agent in self.possible_agents}
            for key in ["hl", "ll"]
        }

    def discount_spec(self):
        return {
            key: {  # TODO (sasha) should rather convert the discounts + rewards to float32/float16
                agent: specs.BoundedArray((), np.float64, minimum=0, maximum=1.0)
                for agent in self.possible_agents
            }
            for key in ["hl", "ll"]
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

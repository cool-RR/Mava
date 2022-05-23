from abc import ABC, abstractmethod

import numpy as np
from dm_env._environment import TimeStep

from mava.wrappers.env_wrappers import ParallelEnvWrapper


class HrlEnvironmentWrapper(ParallelEnvWrapper, ABC):
    """
    A wrapper for environments using HRL algorithms in mava.

    `step` and `reset` must return a single timestep for the higher level agent. The lower level
    agent receives its timestep from the `ll_timestep` method.

    all specs should be formatted as:
    {
      hl: {agent_0: spec, ...},
      ll: {agent_0: spec, ...}
    }
    so that each agent at each level gets a spec

    where hl stands for high level (agent) and ll stands for low level (agent)
    """

    @abstractmethod
    def ll_timestep(self, ts: TimeStep, hl_actions: np.ndarray) -> TimeStep:
        """
        Get the observations for the lower level agent given the higher level observations
        and actions.
        """
        pass

    # Specs
    @abstractmethod
    def observation_spec(self):
        pass

    @abstractmethod
    def action_spec(self):
        pass

    @abstractmethod
    def reward_spec(self):
        pass

    @abstractmethod
    def discount_spec(self):
        pass

    @abstractmethod
    def extra_spec(self):
        pass

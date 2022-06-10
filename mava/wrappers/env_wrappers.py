# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Any, Iterator, List

import dm_env
import jax.numpy as jnp
from chex import Array

from mava.utils.id_utils import EntityId


class ParallelEnvWrapper(dm_env.Environment):
    """Abstract class for parallel environment wrappers"""

    @abstractmethod
    def env_done(self) -> bool:
        """Returns a bool indicating if env is done"""

    @property
    @abstractmethod
    def agents(self) -> List:
        """
        Returns the active agents in the env.
        """

    @property
    @abstractmethod
    def possible_agents(self) -> List:
        """
        Returns all the possible agents in the env.
        """


class SequentialEnvWrapper(ParallelEnvWrapper):
    """
    Abstract class for sequential environment wrappers.
    """

    @abstractmethod
    def agent_iter(self, max_iter: int) -> Iterator:
        """
        Returns an iterator that yields the current agent in the env.
            max_iter: Maximum number of iterations (to limit infinite loops/iterations).
        """

    @property
    @abstractmethod
    def current_agent(self) -> Any:
        """
        Returns the current selected agent.
        """


class MAMCTSWrapper(ParallelEnvWrapper):
    """Abstract class for environment models used in MAMCTS"""

    @abstractmethod
    def get_observation(self, environment_state, agent_info) -> Array:
        """Returns an agent's observation given the environment state and an agents indentifying information"""

    @abstractmethod
    def get_possible_agents(self) -> List[EntityId]:
        """Returns a list of all agent EntityID objects"""

    def get_agent_mask(self, environment_state, agent_info) -> Array:
        """Return an agent's current action mask - by default all actions are available
        available actions are represented by zeros and invalid actions are represented by ones"""

        return jnp.zeros((self.action_spec()[agent_info].num_values,), dtype=jnp.int32)

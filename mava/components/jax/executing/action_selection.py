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

"""Execution components for system builders"""
import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict

import acme.jax.utils as utils
import chex
import jax
import numpy as np
import jax.numpy as jnp
from acme.jax import utils

from mava.components.jax import Component
from mava.core_jax import SystemExecutor
from mava.systems.jax.mamcts.mcts import MCTS, MaxDepth, RecurrentFn, RootFn, TreeSearch
from mava.utils import tree_utils
from mava.utils.id_utils import EntityId


@dataclass
class ExecutorSelectActionProcessConfig:
    pass


class FeedforwardExecutorSelectAction(Component):
    def __init__(
        self,
        config: ExecutorSelectActionProcessConfig = ExecutorSelectActionProcessConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    # Select actions
    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Summary"""
        executor.store.actions_info = {}
        executor.store.policies_info = {}

        for agent, observation in executor.store.observations.items():
            action_info, policy_info = executor.select_action(agent, observation)
            executor.store.actions_info[agent] = action_info
            executor.store.policies_info[agent] = policy_info

    # Select action
    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        """Summary"""

        agent = executor.store.agent
        network = executor.store.networks["networks"][
            executor.store.agent_net_keys[agent]
        ]

        observation = utils.add_batch_dim(executor.store.observation.observation)

        rng_key, executor.store.key = jax.random.split(executor.store.key)

        # TODO (dries): We are currently using jit in the networks per agent.
        # We can also try jit over all the agents in a for loop. This would
        # allow the jit function to save us even more time.
        executor.store.action_info, executor.store.policy_info = network.get_action(
            observation,
            rng_key,
            utils.add_batch_dim(executor.store.observation.legal_actions),
        )

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "executor_select_action"


@dataclass
class MCTSConfig:
    root_fn: RootFn = None
    recurrent_fn: RecurrentFn = None
    search: TreeSearch = None
    environment_model: Any = None
    num_simulations: int = 10
    evaluator_num_simulations: int = 50
    max_depth: MaxDepth = None
    other_search_params: Callable[[None], Dict[str, Any]] = lambda: {}
    evaluator_other_search_params: Callable[[None], Dict[str, Any]] = lambda: {}


class MCTSFeedforwardExecutorSelectAction(FeedforwardExecutorSelectAction):
    """MCTS action selection"""

    def __init__(
        self,
        config: MCTSConfig = MCTSConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        super().__init__(config)

    def on_execution_init_start(self, executor: SystemExecutor) -> None:

        if None in [self.config.root_fn, self.config.recurrent_fn, self.config.search]:
            raise ValueError("Required arguments for MCTS config have not been given")

        self.mcts = MCTS(self.config)

    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Summary"""
        executor.store.actions_info = {}
        executor.store.policies_info = {}

        num_agents = len(executor.store.observations)
        # Store agent_ids, observations and params in stacked pytrees in the same order
        # ie observations[0] belongs to agent with id agent_ids[0] with params = params[0]
        agent_ids = list(executor.store.observations.keys())
        observations = [
            executor.store.observations[agent_id].observation for agent_id in agent_ids
        ]
        params = [
            executor.store.networks["networks"][
                executor.store.agent_net_keys[agent_id]
            ].params
            for agent_id in agent_ids
        ]
        stacked_agent_ids = tree_utils.stack_trees(agent_ids)
        params = tree_utils.stack_trees(params)
        observations = tree_utils.stack_trees(observations)

        # Selecting the first net function from networks, this assumes that all agents have the
        # same network. It seems as it is not possible to do this for agents with different networks
        # as functions are not jittable and cannot be put into jnp arrays
        net = executor.store.networks["networks"][
            executor.store.agent_net_keys[EntityId.first()]
        ].network

        def forward_fn(observations, params, key):
            return net.apply(params, observations)

        action_infos, policy_infos = jax.vmap(
            functools.partial(
                self.vmappable_select_action,
                forward_fn=forward_fn,
                rng_key=jax.random.PRNGKey(0),
                executor=executor,
            )
        )(params=params, observation=observations, agent=stacked_agent_ids)

        for agent_id in agent_ids:
            i = agent_id.index(num_agents)
            # TODO (sasha): should this also be `index_stacked_tree`?
            executor.store.actions_info[agent_id] = action_infos[i]
            executor.store.policies_info[agent_id] = tree_utils.index_stacked_tree(
                policy_infos, i
            )

    def vmappable_select_action(
        self, params, forward_fn, observation, rng_key, executor, agent
    ):
        observation = utils.add_batch_dim(observation)

        return self.mcts.get_action(
            forward_fn,
            params,
            rng_key,
            executor.store.environment_state,
            observation,
            agent,
            executor.store.is_evaluator,
        )

    # # Select action
    # def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
    #     """Summary"""
    #
    #     agent = executor.store.agent
    #     network = executor.store.networks["networks"][
    #         executor.store.agent_net_keys[agent]
    #     ]
    #
    #     rng_key, executor.store.key = jax.random.split(executor.store.key)
    #
    #     observation = utils.add_batch_dim(executor.store.observation.observation)
    #
    #     executor.store.action_info, executor.store.policy_info = self.mcts.get_action(
    #         network.forward_fn,
    #         network.params,
    #         rng_key,
    #         executor.store.environment_state,
    #         observation,
    #         agent,
    #         executor.store.is_evaluator,
    #     )

    @staticmethod
    def config_class() -> Callable:
        return MCTSConfig

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

"""Jax MAMCTS system networks."""
import dataclasses
import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from dm_env import specs as dm_specs
from jax import jit

from mava import specs as mava_specs

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


@dataclasses.dataclass
class MAMCTSNetworks:
    """TODO: Add description here."""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
    ) -> None:
        """TODO: Add description here."""
        self.network = network
        self.params = params

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            logits, value = self.network.apply(params, observations)

            return logits, value

        self.forward_fn = forward_fn

    def get_logits(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        logits, _ = self.forward_fn(self.params, observations)

        return logits

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        _, value = self.forward_fn(self.params, observations)
        return value

    def get_logits_and_value(
        self, observations: networks_lib.Observation
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """TODO: Add description here."""
        logits, value = self.forward_fn(self.params, observations)
        return logits, value


def make_mcts_network(
    network: networks_lib.FeedForwardNetwork,
    params: Dict[str, jnp.ndarray],
) -> MAMCTSNetworks:
    """TODO: Add description here."""
    return MAMCTSNetworks(
        network=network,
        params=params,
    )


def make_networks(
    spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    observation_network=utils.batch_concat,
) -> MAMCTSNetworks:
    """TODO: Add description here."""
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_networks(
            environment_spec=spec,
            key=key,
            base_layer_sizes=policy_layer_sizes,
            observation_network=observation_network,
        )
    else:
        raise NotImplementedError(
            "Continuous networks not implemented yet."
            + "See: https://github.com/deepmind/acme/blob/"
            + "master/acme/agents/jax/ppo/networks.py"
        )


def make_discrete_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    base_layer_sizes: Sequence[int],
    observation_network=utils.batch_concat,
) -> MAMCTSNetworks:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.
    def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        policy_value_network = hk.Sequential(
            [
                observation_network,
                hk.nets.MLP(base_layer_sizes, activation=jax.nn.relu),
                networks_lib.CategoricalValueHead(num_values=num_actions),
            ]
        )
        return policy_value_network(inputs)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return make_mcts_network(
        network=forward_fn,
        params=params,
    )


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    observation_network=utils.batch_concat,
) -> Dict[str, Any]:
    """Description here"""

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_networks(
            specs[net_key],
            key=rng_key,
            policy_layer_sizes=policy_layer_sizes,
            observation_network=observation_network,
        )

    return {
        "networks": networks,
    }


@dataclasses.dataclass
class RepresentationNetwork:
    """TODO: Add description here."""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
    ) -> None:
        """TODO: Add description here."""
        self.network = network
        self.params = params

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            observation_history: networks_lib.Observation,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            root_embedding = self.network.apply(params, observation_history)
            return root_embedding

        self.forward_fn = forward_fn


def make_representation_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    base_layer_sizes: Sequence[int],
    observation_network=utils.batch_concat,
) -> RepresentationNetwork:
    """TODO: Add description here."""

    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.
    def forward_fn(observation_history: jnp.ndarray) -> networks_lib.FeedForwardNetwork:

        representation_network = hk.Sequential(
            [
                observation_network,
                hk.nets.MLP(base_layer_sizes, activation=jax.nn.relu),
            ]
        )

        return representation_network(observation_history)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return make_representation_network(
        network=forward_fn,
        params=params,
    )


def make_representation_network(
    network: networks_lib.FeedForwardNetwork,
    params: Dict[str, jnp.ndarray],
) -> RepresentationNetwork:
    """TODO: Add description here."""
    return RepresentationNetwork(
        network=network,
        params=params,
    )


@dataclasses.dataclass
class EnvironmentDynamicsModel:
    """TODO: Add description here."""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
    ) -> None:
        """TODO: Add description here."""
        self.network = network
        self.params = params

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            previous_embedding: networks_lib.NetworkOutput,
            action: networks_lib.Action,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            reward, embedding = self.network.apply(params, previous_embedding, action)

            return reward, embedding

        self.forward_fn = forward_fn


def make_environment_model_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    base_layer_sizes: Sequence[int],
    embedding_head_layer_sizes: Sequence[int],
    reward_head_layer_sizes: Sequence[int],
    action_embedding_size: int,
    observation_network=utils.batch_concat,
) -> EnvironmentDynamicsModel:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.
    def forward_fn(
        prev_embedding: jnp.ndarray, action: jnp.ndarray
    ) -> networks_lib.FeedForwardNetwork:

        action_embedding_network = hk.Embed(num_actions, action_embedding_size)

        embedded_action = action_embedding_network(action)

        inputs = jnp.concatenate((prev_embedding, embedded_action), -1)

        base_dynamics_network = hk.Sequential(
            [
                hk.nets.MLP(base_layer_sizes, activation=jax.nn.relu),
            ]
        )
        embedding_head = hk.nets.MLP(embedding_head_layer_sizes, activation=jax.nn.relu)

        reward_head = hk.nets.MLP(reward_head_layer_sizes, activation=jax.nn.relu)

        inter_state = base_dynamics_network(inputs)

        embedding = embedding_head(inter_state)
        reward = reward_head(inter_state)

        return embedding, reward

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_embedding = jnp.zeros(embedding_head_layer_sizes[-1])
    dummy_embedding = utils.add_batch_dim(dummy_embedding)  # Dummy 'sequence' dim.
    dummy_action = jnp.zeros((), int)
    dummy_action = utils.add_batch_dim(dummy_action)

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_embedding, dummy_action)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return make_environment_network(
        network=forward_fn,
        params=params,
    )


def make_environment_network(
    network: networks_lib.FeedForwardNetwork,
    params: Dict[str, jnp.ndarray],
) -> EnvironmentDynamicsModel:
    """TODO: Add description here."""
    return EnvironmentDynamicsModel(
        network=network,
        params=params,
    )


def make_policy_value_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    base_layer_sizes: Sequence[int],
    learned_model_embedding_size: int = 10,
    observation_network=utils.batch_concat,
) -> MAMCTSNetworks:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.
    def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        policy_value_network = hk.Sequential(
            [
                observation_network,
                hk.nets.MLP(base_layer_sizes, activation=jax.nn.relu),
                networks_lib.CategoricalValueHead(num_values=num_actions),
            ]
        )
        return policy_value_network(inputs)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = jnp.zeros(learned_model_embedding_size)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return make_mcts_network(
        network=forward_fn,
        params=params,
    )


class LearnedModelNetworks:
    """TODO: Add description here."""

    def __init__(
        self,
        mamcts_network: MAMCTSNetworks,
        environment_dynamics_network: EnvironmentDynamicsModel,
        representation_network: RepresentationNetwork,
    ) -> None:
        """TODO: Add description here."""
        self.policy_value_network = mamcts_network
        self.dynamics = environment_dynamics_network
        self.representation = representation_network
        self.params = {
            "policy": self.policy_value_network.params,
            "dynamics": self.dynamics.params,
            "representation": self.representation.params,
        }

        self.policy_value_fn = self.policy_value_network.forward_fn
        self.dynamics_fn = self.dynamics.forward_fn
        self.representation_fn = self.representation.forward_fn

    def get_policy_value(self, embedding):
        return self.policy_value_fn(self.params["policy"], embedding)

    def get_next_state_reward(self, embedding, action):
        return self.dynamics_fn(self.params["dynamics"], embedding, action)

    def get_root_state(self, observation_history):
        return self.representation_fn(
            self.params["representation"], observation_history
        )


def make_learned_model_networks(
    spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    base_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    embedding_head_layer_sizes: Sequence[int] = (256, 100),
    reward_head_layer_sizes: Sequence[int] = (256, 100),
    action_embedding_size: int = 10,
    observation_network=utils.batch_concat,
) -> LearnedModelNetworks:
    """TODO: Add description here."""

    mamcts_net = make_policy_value_networks(
        environment_spec=spec,
        key=key,
        base_layer_sizes=base_layer_sizes,
        learned_model_embedding_size=embedding_head_layer_sizes[-1],
    )

    environment_dynamics_net = make_environment_model_networks(
        environment_spec=spec,
        key=key,
        base_layer_sizes=base_layer_sizes,
        action_embedding_size=action_embedding_size,
        embedding_head_layer_sizes=embedding_head_layer_sizes,
        reward_head_layer_sizes=reward_head_layer_sizes,
        observation_network=observation_network,
    )

    base_layer_sizes = (*base_layer_sizes, embedding_head_layer_sizes[-1])
    representation_net = make_representation_networks(
        environment_spec=spec,
        key=key,
        base_layer_sizes=base_layer_sizes,
        observation_network=observation_network,
    )

    return LearnedModelNetworks(
        mamcts_network=mamcts_net,
        environment_dynamics_network=environment_dynamics_net,
        representation_network=representation_net,
    )


def make_default_learned_model_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    base_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    embedding_head_layer_sizes: Sequence[int] = (256, 100),
    reward_head_layer_sizes: Sequence[int] = (256, 100),
    action_embedding_size: int = 10,
    observation_network=utils.batch_concat,
) -> Dict[str, Any]:
    """Description here"""

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_learned_model_networks(
            specs[net_key],
            key=rng_key,
            base_layer_sizes=base_layer_sizes,
            embedding_head_layer_sizes=embedding_head_layer_sizes,
            reward_head_layer_sizes=reward_head_layer_sizes,
            action_embedding_size=action_embedding_size,
            observation_network=observation_network,
        )

    return {
        "networks": networks,
    }

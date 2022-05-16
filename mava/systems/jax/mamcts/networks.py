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
from haiku import MultiHeadAttention
from haiku._src.basic import merge_leading_dims
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
            search_tree,
            messages,
            key: networks_lib.PRNGKey,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """TODO: Add description here."""
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            (logits, value), message = self.network.apply(
                params, observations, search_tree, messages
            )

            return logits, value, message

        self.forward_fn = forward_fn

    def get_logits(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        logits, _, _ = self.forward_fn(self.params, observations)

        return logits

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """TODO: Add description here."""
        _, value, _ = self.forward_fn(self.params, observations)
        return value

    def get_logits_and_value(
        self, observations: networks_lib.Observation
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """TODO: Add description here."""
        logits, value, _ = self.forward_fn(self.params, observations)
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
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    observation_network=utils.batch_concat,
) -> MAMCTSNetworks:
    """TODO: Add description here."""
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_networks(
            environment_spec=spec,
            key=key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
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
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    observation_network=utils.batch_concat,
    num_heads: int = 1,
    key_dim=10,
    message_size=10,
) -> MAMCTSNetworks:
    """TODO: Add description here."""

    num_actions = environment_spec.actions.num_values

    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.
    def forward_fn(
        observations: jnp.ndarray, search_tree: jnp.ndarray, messages: jnp.ndarray
    ) -> networks_lib.FeedForwardNetwork:
        policy_value_network = hk.Sequential(
            [
                hk.nets.MLP(policy_layer_sizes, activation=jax.nn.relu),
                networks_lib.CategoricalValueHead(num_values=num_actions),
            ]
        )

        attention_network = hk.MultiHeadAttention(
            num_heads, key_dim, 1.0, model_size=message_size
        )
        search_tree_shape = search_tree.shape

        search_trees = merge_leading_dims(search_tree, 2)
        processed_tree = observation_network(search_trees)
        processed_tree = processed_tree.reshape(*search_tree_shape[0:2], -1)

        new_message = attention_network(
            query=jnp.expand_dims(messages, -2),
            key=processed_tree,
            value=processed_tree,
        )

        obs_out = observation_network(observations)
        msg_concat = jnp.concatenate([obs_out, messages], -1)

        return policy_value_network(msg_concat), new_message

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))
    # TODO Change dummys
    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_message = jnp.zeros(key_dim)
    dummy_tree = jnp.zeros((1, *environment_spec.observations.observation.shape))

    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
    dummy_message = utils.add_batch_dim(dummy_message)
    dummy_tree = utils.add_batch_dim(dummy_tree)

    network_key, key = jax.random.split(key)
    params = forward_fn.init(network_key, dummy_obs, dummy_tree, dummy_message)  # type: ignore

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
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
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
            critic_layer_sizes=critic_layer_sizes,
            observation_network=observation_network,
        )

    return {
        "networks": networks,
    }

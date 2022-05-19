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

"""Jax MAPPO system networks."""
from typing import Any, Callable, Dict, List, Sequence, Tuple

import haiku as hk  # type: ignore
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from dm_env import specs as dm_specs

from mava import specs as mava_specs
from mava.systems.jax.hrl.hrl_env_spec import invert_hrl_spec_dict
from mava.systems.jax.mappo.networks import PPONetworks, make_discrete_networks

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


def make_networks(
    spec: Dict[str, specs.EnvironmentSpec],
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
) -> Dict[str, PPONetworks]:
    """TODO: Add description here."""
    if isinstance(spec["hl"].actions, specs.DiscreteArray) and isinstance(
        spec["ll"].actions, specs.DiscreteArray
    ):
        return {
            "hl": make_discrete_networks(
                environment_spec=spec["hl"],
                key=key,
                policy_layer_sizes=policy_layer_sizes,
                critic_layer_sizes=critic_layer_sizes,
            ),
            "ll": make_discrete_networks(
                environment_spec=spec["ll"],
                key=key,
                policy_layer_sizes=policy_layer_sizes,
                critic_layer_sizes=critic_layer_sizes,
            ),
        }
    else:
        raise NotImplementedError(
            "Continuous networks not implemented yet."
            + "See: https://github.com/deepmind/acme/blob/"
            + "master/acme/agents/jax/ppo/networks.py"
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
        )

    return {
        "networks": networks,
    }

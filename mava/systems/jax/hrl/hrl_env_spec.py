from typing import Dict

import dm_env
from acme.specs import EnvironmentSpec

from mava import MAEnvironmentSpec, specs
from mava.components.jax import building
from mava.components.jax.building.environments import EnvironmentSpecConfig
from mava.core_jax import SystemBuilder
from mava.utils.sort_utils import sort_str_num


class HrlMAEnvironmentSpec(MAEnvironmentSpec):
    HIGH_LEVEL = "hl"
    LOW_LEVEL = "ll"

    def _make_ma_environment_spec(
        self, environment: dm_env.Environment  # TODO type as hrl env wrapper
    ) -> Dict[str, Dict[str, EnvironmentSpec]]:
        """
        Returns dict of dicts of EnvironmentSpecs for each agent's high level and low level policy
        """

        specs = {}
        observation_specs = environment.observation_spec()
        action_specs = environment.action_spec()
        reward_specs = environment.reward_spec()
        discount_specs = environment.discount_spec()
        self.extra_specs = environment.extra_spec()

        print(f"hrl ma env spec got: {observation_specs}")

        for agent in environment.possible_agents:
            specs[agent] = {
                self.HIGH_LEVEL: EnvironmentSpec(
                    observations=observation_specs[self.HIGH_LEVEL][agent],
                    actions=action_specs[self.HIGH_LEVEL][agent],
                    rewards=reward_specs[self.HIGH_LEVEL][agent],
                    discounts=discount_specs[self.HIGH_LEVEL][agent],
                ),
                self.LOW_LEVEL: EnvironmentSpec(
                    observations=observation_specs[self.LOW_LEVEL][agent],
                    actions=action_specs[self.LOW_LEVEL][agent],
                    rewards=reward_specs[self.LOW_LEVEL][agent],
                    discounts=discount_specs[self.LOW_LEVEL][agent],
                ),
            }

        return specs

    def get_agent_type_specs(self):
        raise NotImplementedError

    def get_agent_types(self):
        raise NotImplementedError


class HrlEnvironmentSpec(building.EnvironmentSpec):
    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """[summary]"""

        builder.store.environment_spec = HrlMAEnvironmentSpec(
            self.config.environment_factory()
        )

        builder.store.agents = sort_str_num(
            builder.store.environment_spec.get_agent_ids()
        )
        builder.store.extras_spec = {}


def invert_hrl_spec_dict(specs):
    """
    Inverts an HrlMAEnvironmentSpec dict from having agent as the first key to having high/low
    level as the first key

    """
    HIGH_LEVEL = HrlMAEnvironmentSpec.HIGH_LEVEL
    LOW_LEVEL = HrlMAEnvironmentSpec.LOW_LEVEL

    new_specs = {
        HIGH_LEVEL: {},
        LOW_LEVEL: {},
    }

    print(specs)

    for agent_id, spec in specs.items():
        new_specs[HIGH_LEVEL][agent_id] = spec[HIGH_LEVEL]
        new_specs[LOW_LEVEL][agent_id] = spec[LOW_LEVEL]

    return new_specs

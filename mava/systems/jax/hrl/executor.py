from types import SimpleNamespace
from typing import Dict, List

import dm_env

from mava.callbacks import Callback
from mava.systems.jax import Executor
from mava.types import NestedArray


class HrlExecutor(Executor):
    def __init__(self, config: SimpleNamespace, components: List[Callback] = []):
        super().__init__(config, components)

        self.adders = self.store.adder
        self.networks = self.store.networks

        # TODO (sasha) make it a dict of adder for consistency
        if not self._evaluator:
            assert isinstance(
                self.adders, list
            ), f"expected tuple of adders, but got {type(self.adders)}"

        # assert isinstance(
        #     self.networks, list
        # ), f"expected tuple of networks, but got {type(self.networks)}"

    def _use_hl_adder(self):
        if not self._evaluator:
            self.store.adder = self.adders[0]

    def _use_ll_adder(self):
        if not self._evaluator:
            self.store.adder = self.adders[1]

    def _set_network_level(self, level):
        self.store.networks = {
            "networks": {agent_id: network[level]}
            for agent_id, network in self.networks["networks"].items()
        }

    def _use_hl_networks(self):
        self._set_network_level("hl")

    def _use_ll_networks(self):
        self._set_network_level("ll")

    def hl_observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, NestedArray] = {},
    ) -> None:
        self._use_hl_adder()
        return self.observe_first(timestep, extras)

    def ll_observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, NestedArray] = {},
    ) -> None:
        self._use_ll_adder()
        return self.observe_first(timestep, extras)

    def hl_observe(
        self,
        actions: Dict[str, NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, NestedArray] = {},
    ) -> None:
        self._use_hl_adder()
        self.observe(actions, next_timestep, next_extras)
        # for safety to make sure we always select an adder level before observing
        self.store.adder = self.adders

    def ll_observe(
        self,
        actions: Dict[str, NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, NestedArray] = {},
    ) -> None:
        self._use_ll_adder()
        self.observe(actions, next_timestep, next_extras)
        # for safety to make sure we always select an adder level before observing
        self.store.adder = self.adders

    def select_hl_actions(self, observation: NestedArray) -> NestedArray:
        self._use_hl_networks()
        actions = self.select_actions(observation)
        # for safety to make sure we always select a network level before selecting actions
        self.store.networks = self.networks
        return actions

    def select_ll_actions(self, observation: NestedArray) -> NestedArray:
        self._use_ll_networks()
        actions = self.select_actions(observation)
        # for safety to make sure we always select a network level before selecting actions
        self.store.networks = self.networks
        return actions

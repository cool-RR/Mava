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
        print(self.adders)
        print(self.networks)

        assert isinstance(
            self.adders, list
        ), f"expected tuple of adders, but got {type(self.adders)}"

        # assert isinstance(
        #     self.networks, list
        # ), f"expected tuple of networks, but got {type(self.networks)}"

    def use_hl_adder(self):
        self.store.adder = self.adders[0]

    def use_ll_adder(self):
        self.store.adder = self.adders[1]

    def hl_observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, NestedArray] = {},
    ) -> None:
        self.use_hl_adder()
        return self.observe_first(timestep, extras)

    def ll_observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, NestedArray] = {},
    ) -> None:
        self.use_ll_adder()
        return self.observe_first(timestep, extras)

    def hl_observe(
        self,
        actions: Dict[str, NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, NestedArray] = {},
    ) -> None:
        self.use_hl_adder()
        return self.observe(actions, next_timestep, next_extras)

    def ll_observe(
        self,
        actions: Dict[str, NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, NestedArray] = {},
    ) -> None:
        self.use_ll_adder()
        return self.observe(actions, next_timestep, next_extras)

    def select_hl_actions(
        self,
        agent: str,
        observation: NestedArray,
        state: NestedArray = None,
    ) -> NestedArray:
        pass

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

# TODO (Arnu): reintroduce proper return types, e.g. data/parameter server once those
# have been created.

"""Jax-based Mava system builder implementation."""

from typing import Any, List

from mava.callbacks import Callback
from mava.systems.jax import Builder
from mava.systems.jax.hrl.executor import HrlExecutor


class HrlBuilder(Builder):
    def __init__(self, components: List[Callback]) -> None:
        """System building init

        Args:
            components: system callback component
        """
        super().__init__(components)

    def executor(
        self, executor_id: str, data_server_client: Any, parameter_server_client: Any
    ) -> Any:
        """Executor, a collection of agents in an environment to gather experience.

        Args:
            executor_id : id to identify the executor process for logging purposes
            data_server_client : data server client for pushing transition data
            parameter_server_client : parameter server client for pulling parameters
        Returns:
            System executor
        """
        print(f"[hrl builder]: dataserver: {type(data_server_client)}")
        self.store.executor_id = executor_id
        self.store.data_server_client = data_server_client
        self.store.parameter_server_client = parameter_server_client
        self.store.is_evaluator = self.store.executor_id == "evaluator"

        # start of making the executor
        self.on_building_executor_start()

        # make adder if required
        if not self.store.is_evaluator:
            # make adder signature
            self.on_building_executor_adder_priority()

            # make rate limiter
            self.on_building_executor_adder()
        else:
            self.store.adder = None

        # make executor logger
        self.on_building_executor_logger()

        # make executor parameter client
        self.on_building_executor_parameter_client()

        # make executor
        self.on_building_executor()

        # create the executor
        self.store.executor = HrlExecutor(
            config=self.store,
            components=self.callbacks,
        )

        # make copy of environment
        self.on_building_executor_environment()

        # make train loop
        self.on_building_executor_environment_loop()

        # end of making the executor
        self.on_building_executor_end()

        # return the environment loop
        return self.store.system_executor

    def build(self) -> None:
        """Construct program nodes."""

        # start of system building
        self.on_building_start()

        # build program nodes
        self.on_building_program_nodes()

        # end of system building
        self.on_building_end()

    def launch(self) -> None:
        """Run the graph program."""

        # start of system launch
        self.on_building_launch_start()

        # launch system
        self.on_building_launch()

        # end of system launch
        self.on_building_launch_end()

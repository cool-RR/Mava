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

# TODO(Arnu): remove at a later stage
# type: ignore

"""Tests for config class for Jax-based Mava systems"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List

import pytest

from mava.callbacks import Callback
from mava.core_jax import SystemBuilder
from mava.systems.jax.system import System


# Mock components to feed to the builder
@dataclass
class MockDataServerAdderDefaultConfig:
    adder_param_0: int = 1
    adder_param_1: str = "1"


class MockDataServerAdder(Callback):
    def __init__(
        self,
        config: MockDataServerAdderDefaultConfig = MockDataServerAdderDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.adder_signature = {"adder_signature": self.config.adder_param_0}

    def on_building_data_server_rate_limiter(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.rate_limiter = {"rate_limiter": self.config.adder_param_1}

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "data_server_adder"


@dataclass
class MockDataServerDefaultConfig:
    data_server_param_0: int = 1
    data_server_param_1: str = "1"


class MockDataServer(Callback):
    def __init__(
        self, config: MockDataServerDefaultConfig = MockDataServerDefaultConfig()
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.system_data_server = {
            "data_server": (
                builder.adder_signature,
                builder.rate_limiter,
                self.config.data_server_param_0,
                self.config.data_server_param_1,
            )
        }

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "data_server"


@dataclass
class MockParameterServerDefaultConfig:
    parameter_server_param_0: int = 1
    parameter_server_param_1: str = "1"


class MockParameterServer(Callback):
    def __init__(
        self,
        config: MockParameterServerDefaultConfig = MockParameterServerDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_parameter_server(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.system_parameter_server = {
            "parameter_server": (
                self.config.parameter_server_param_0,
                self.config.parameter_server_param_1,
            )
        }

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "parameter_server"


@dataclass
class MockExecutorAdderDefaultConfig:
    executor_adder_param_0: int = 1
    executor_adder_param_1: str = "1"


class MockExecutorAdder(Callback):
    def __init__(
        self,
        config: MockExecutorAdderDefaultConfig = MockExecutorAdderDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_adder_priority(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.adder_priority = {"adder_priority": self.config.executor_adder_param_0}

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.adder = {
            "adder": (builder.adder_priority, self.config.executor_adder_param_1)
        }

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor_adder"


@dataclass
class MockExecutorDefaultConfig:
    executor_param_0: int = 1
    executor_param_1: str = "1"


class MockExecutor(Callback):
    def __init__(
        self,
        config: MockExecutorDefaultConfig = MockExecutorDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.exec_logger = {"exec_logger": self.config.executor_param_0}

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.exec_param_client = {"exec_param_client": self.config.executor_param_1}

    def on_building_executor(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.exec = {"exec": (builder.adder, builder.exec_param_client)}

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.env = {"env": builder.exec_logger}

    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.system_executor = {"executor": (builder.env, builder.exec)}

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor"


@dataclass
class MockTrainerDatasetDefaultConfig:
    trainer_dataset_param_0: int = 1


class MockTrainerDataset(Callback):
    def __init__(
        self,
        config: MockTrainerDatasetDefaultConfig = MockTrainerDatasetDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.dataset = {"dataset": self.config.trainer_dataset_param_0}

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer_dataset"


@dataclass
class MockTrainerDefaultConfig:
    trainer_param_0: int = 1
    trainer_param_1: str = "1"


class MockTrainer(Callback):
    def __init__(
        self,
        config: MockTrainerDefaultConfig = MockTrainerDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.train_logger = {"train_logger": self.config.trainer_param_0}

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.train_param_client = {"train_param_client": self.config.trainer_param_1}

    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.system_trainer = {
            "trainer": (builder.train_logger, builder.train_param_client)
        }

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer"


@dataclass
class DistributorDefaultConfig:
    num_executors: int = 1
    nodes_on_gpu: List[str] = field(default_factory=list)
    multi_process: bool = True
    name: str = "system"


class MockDistributor(Callback):
    def __init__(
        self, config: DistributorDefaultConfig = DistributorDefaultConfig()
    ) -> None:
        """Mock system distributor component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "distributor"


@dataclass
class MockProgramDefaultConfig:
    program_param_0: int = 1
    program_param_1: str = "1"


class MockProgramConstructor(Callback):
    def __init__(
        self,
        config: MockProgramDefaultConfig = MockProgramDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.data_server()
        builder.parameter_server()
        builder.executor()
        builder.trainer()
        builder.system_build = {
            "system": (
                builder.system_executor,
                builder.system_trainer,
                builder.system_data_server,
                builder.system_parameter_server,
                self.config.program_param_0,
                self.config.program_param_1,
            )
        }

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "program"


@dataclass
class MockLauncherDefaultConfig:
    launcher_param_0: int = 1
    launcher_param_1: str = "1"


class MockLauncher(Callback):
    def __init__(
        self,
        config: MockLauncherDefaultConfig = MockLauncherDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_launch(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.system_launcher = {
            "launcher": (
                builder.system_build,
                self.config.launcher_param_0,
                self.config.launcher_param_1,
            )
        }

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "launcher"


class TestSystem(System):
    def design(self) -> SimpleNamespace:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = SimpleNamespace(
            data_server_adder=MockDataServerAdder,
            data_server=MockDataServer,
            parameter_server=MockParameterServer,
            executor_adder=MockExecutorAdder,
            executor=MockExecutor,
            trainer_dataset=MockTrainerDataset,
            trainer=MockTrainer,
            distributor=MockDistributor,
            program=MockProgramConstructor,
            launcher=MockLauncher,
        )
        return components


@pytest.fixture
def test_system() -> System:
    """Dummy system with zero components."""
    return TestSystem()


def test_builder(
    test_system: System,
) -> None:
    """Test if system can launch without having had changed (configured) the default \
        config."""
    test_system.launch(num_executors=1, nodes_on_gpu=["process"])

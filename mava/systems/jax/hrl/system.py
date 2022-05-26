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

"""Jax Hrl system."""
import copy
from typing import Any, Tuple

from mava.components.jax import building, executing, training, updating
from mava.specs import DesignSpec
from mava.systems.jax import System
from mava.systems.jax.hrl.adder import HrlParallelSequenceAdder
from mava.systems.jax.hrl import components
from mava.systems.jax.hrl.env_loop import HrlParallelExecutorEnvironmentLoop
from mava.systems.jax.hrl.hrl_builder import HrlBuilder
from mava.systems.jax.hrl.hrl_distributor import HrlDistributor
from mava.systems.jax.hrl.hrl_env_spec import HrlEnvironmentSpec
from mava.systems.jax.mappo.components import ExtrasLogProbSpec
from mava.systems.jax.mappo.config import MAPPODefaultConfig


class HrlSystem(System):
    def design(self) -> Tuple[DesignSpec, Any]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        # Set the default configs
        default_params = MAPPODefaultConfig()

        # Default system processes
        # System initialization
        system_init = DesignSpec(
            environment_spec=HrlEnvironmentSpec, system_init=building.SystemInit
        ).get()

        # Executor
        executor_process = DesignSpec(
            executor_init=executing.ExecutorInit,
            executor_observe=executing.FeedforwardExecutorObserve,
            executor_select_action=executing.FeedforwardExecutorSelectAction,
            executor_adder=HrlParallelSequenceAdder,
            executor_environment_loop=HrlParallelExecutorEnvironmentLoop,
            networks=building.DefaultNetworks,
        ).get()

        # Trainer
        trainer_process = DesignSpec(
            trainer_init=training.TrainerInit,
            gae_fn=training.GAE,
            loss=training.MAPGWithTrustRegionClippingLoss,
            epoch_update=training.MAPGEpochUpdate,
            minibatch_update=training.MAPGMinibatchUpdate,
            sgd_step=training.MAPGWithTrustRegionStep,
            step=training.DefaultStep,
            trainer_dataset=building.TrajectoryDataset,
        ).get()

        # Data Server
        data_server_process = DesignSpec(
            data_server=building.OnPolicyDataServer,
            data_server_adder_signature=building.ParallelSequenceAdderSignature,
            extras_spec=ExtrasLogProbSpec,
        ).get()

        # Parameter Server
        parameter_server_process = DesignSpec(
            parameter_server=components.HrlParameterServer,
            executor_parameter_client=components.HrlExecutorParameterClient,
            trainer_parameter_client=components.HrlTrainerParameterClient,
        ).get()

        system = DesignSpec(
            **system_init,
            **data_server_process,
            **parameter_server_process,
            **executor_process,
            **trainer_process,
            distributor=HrlDistributor,
            logger=building.Logger,
        )
        return system, default_params

    def build(self, **kwargs: Any) -> None:
        """Configure system hyperparameters."""

        if self._built:
            raise Exception("System already built.")

        # Add the system defaults, but allow the kwargs to overwrite them.
        if self._default_params:
            parameter = copy.copy(self._default_params.__dict__)
        else:
            parameter = {}
        parameter.update(kwargs)

        self.config.build()

        self.config.set_parameters(**parameter)

        # get system config to feed to component list to update hyperparameter settings
        system_config = self.config.get()

        # update default system component configs
        assert len(self.components) == 0
        for component in self._design.get().values():
            self.components.append(component(system_config))

        # Build system
        self._builder = HrlBuilder(components=self.components)
        self._builder.build()
        self._built = True

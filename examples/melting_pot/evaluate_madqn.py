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


import functools
import random
from datetime import datetime
from typing import Any

import launchpad as lp
import sonnet as snt
from absl import app, flags

from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.environments.meltingpot_utils.env_utils import (
    EnvironmentFactory,
    scenarios_for_substrate,
)
from mava.utils.environments.meltingpot_utils.evaluation_utils import ScenarioEvaluation
from mava.utils.environments.meltingpot_utils.registry import get_system_creator_cls
from mava.utils.environments.meltingpot_utils.registry.base import get_networks_restorer
from mava.utils.environments.meltingpot_utils.training import SubstrateTraining
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")
flags.DEFINE_string("substrate", "clean_up", "substrate to train on.")


def train_on_substrate(checkpoint_dir) -> None:
    """Trains on the specified substrate"""
    # Substrate Environment.
    substrate_environment_factory = EnvironmentFactory(substrate=FLAGS.substrate)

    # Networks.
    network_factory = lp_utils.partial_kwargs(madqn.make_default_networks)

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=FLAGS.mava_id,
        time_delta=log_every,
    )
    system_creator = get_system_creator_cls("MADQN")
    substrate_system_creator = system_creator(
        substrate_environment_factory, network_factory, logger_factory, checkpoint_dir
    )
    substrate_training = SubstrateTraining(substrate_system_creator)

    # train on susbtrate
    substrate_training.run()


def test_on_scenarios(substrate, checkpoint_dir) -> None:
    """Tests the system on all the scenarios associated with the specified substrate"""
    scenarios = scenarios_for_substrate(FLAGS.substrate)
    for scenario in scenarios:
        test_on_scenario(substrate, scenario, checkpoint_dir)


def test_on_scenario(substrate, scenario_name: str, checkpoint_dir: str) -> None:
    """Tests the system on a given scenario"""

    # Scenario Environment.
    scenario_environment_factory = EnvironmentFactory(scenario=scenario_name)

    # Networks.
    network_factory = lp_utils.partial_kwargs(madqn.make_default_networks)

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=FLAGS.mava_id,
        time_delta=log_every,
    )

    system_creator_cls = get_system_creator_cls("MADQN")
    scenario_system_creator = system_creator_cls(
        scenario_environment_factory, network_factory, logger_factory, None
    )
    trained_networks_restorer = get_networks_restorer("MADQN")
    trained_networks = trained_networks_restorer(
        substrate, network_factory, checkpoint_dir
    )

    evaluation_loop = ScenarioEvaluation(
        "MADQN", scenario_system_creator, trained_networks
    )
    evaluation_loop.run()


def main(_: Any) -> None:
    """Train on substrate

    Args:
        _ (Any): ...
    """
    # Checkpointer appends "Checkpoints" to checkpoint_dir
    # checkpoint_dir = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # restore agents
    checkpoint_dir = "/home/app/mava/logs/2021-10-25 08:41:32.632667"

    train_on_substrate(checkpoint_dir)

    # test_on_scenarios(FLAGS.substrate, checkpoint_dir)


if __name__ == "__main__":
    app.run(main)

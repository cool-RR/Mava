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

from datetime import datetime
from typing import Any, Callable
from mava.systems.tf import value_decomposition
from helpers import qmix_agent_network_setter, qmix_evaluation_loop_creator
from helpers import get_trained_qmix_networks

from absl import app, flags

from mava import specs as mava_specs
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)

from mava.utils import loggers, lp_utils
from mava.utils.environments.meltingpot_utils.env_utils import (
    MeltingPotEnvironmentFactory,
    scenarios_for_substrate,
)
from mava.utils.environments.meltingpot_utils.evaluation_utils import (
    AgentNetworks,
    ScenarioEvaluation,
)
from mava.utils.environments.meltingpot_utils.network_utils import (
    make_default_qmix_networks,
)
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("logdir", "./logs", "Base dir to store experiments.")
flags.DEFINE_string(
    "checkpoint_dir", "/home/app/mava/logs/2022-03-16 13:16:11.490222", "directory where checkpoints were saved during training"
)
flags.DEFINE_string("substrate", "clean_up", "scenario to evaluste on")


def evaluate_on_scenarios(substrate: str, checkpoint_dir: str) -> None:
    """Tests the system on all the scenarios associated with the specified substrate

    Args:
        substrate: the name of the substrate for which scenarios would be created
        checkpoint_dir: directory where checkpoint is to be restored from
    """
    scenarios = scenarios_for_substrate(substrate)

    # Networks.
    network_factory = lp_utils.partial_kwargs(make_default_qmix_networks)

    trained_networks = get_trained_qmix_networks(
        substrate, network_factory, checkpoint_dir
    )

    for scenario in scenarios:
        evaluate_on_scenario(scenario, network_factory, trained_networks)


def evaluate_on_scenario(
    scenario_name: str,
    network_factory: Callable[[mava_specs.MAEnvironmentSpec], AgentNetworks],
    trained_networks: AgentNetworks,
) -> None:
    """Evaluates a system on a scenario using already trained networks

    Args:
        scenario_name: name of scenario in which system would be evaluated
        network_factory: for instantiating the agent networks for the system
        trained_networks: agent networks previously trained on the corresponding
            substrate

    """
    # Scenario Environment.
    scenario_environment_factory = MeltingPotEnvironmentFactory(scenario=scenario_name)

    # Log every [log_every] seconds.
    log_every = 10

    def logger_factory(label: str, **kwargs: Any) -> loggers.Logger:
        logger = logger_utils.make_logger(
            scenario_name,
            directory=FLAGS.logdir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=FLAGS.mava_id,
            time_delta=log_every,
        )
        return logger

    # Create qmix system for scenario
    scenario_system = value_decomposition.ValueDecomposition(
        environment_factory=scenario_environment_factory,
        network_factory=network_factory,
        mixer="qmix",
        logger_factory=logger_factory,
        exploration_scheduler_fn=LinearExplorationScheduler(
            epsilon_min=0.05, epsilon_decay=1e-4
        ),
        shared_weights=False,
    )

    # Evaluation loop
    evaluation_loop = ScenarioEvaluation(
        scenario_system,
        qmix_evaluation_loop_creator,
        qmix_agent_network_setter,
        trained_networks,
    )
    evaluation_loop.run()


def main(_: Any) -> None:
    """Evaluate on a scenario

    Args:
        _ (Any): ...
    """
    evaluate_on_scenarios(FLAGS.substrate, FLAGS.checkpoint_dir)


if __name__ == "__main__":
    app.run(main)

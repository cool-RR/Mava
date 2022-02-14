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
from typing import Any, Callable, Dict

import reverb
import sonnet as snt
from absl import app, flags
from acme import specs as acme_specs

from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.core import Executor
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import madqn
from mava.utils import lp_utils
from mava.utils.environments.meltingpot_utils.env_utils import (
    EnvironmentFactory,
    scenarios_for_substrate,
)
from mava.utils.environments.meltingpot_utils.evaluation_utils import (
    AgentNetworks,
    MAVASystem,
    ScenarioEvaluation,
)
from mava.utils.environments.meltingpot_utils.network_utils import (
    make_default_madqn_networks,
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
    "checkpoint_dir", "", "directory where checkpoints were saved during training"
)
flags.DEFINE_string("substrate", "clean_up", "scenario to evaluste on")


def madqn_evaluation_loop_creator(system: MAVASystem) -> ParallelEnvironmentLoop:
    """Creates an environment loop for the evaluation of a system

    Args:
        system ([MAVASystem]): the system to evaluate

    Returns:
        [ParallelEnvironmentLoop]: an environment loop for evaluation
    """
    replay_server = reverb.Server(tables=system.replay())
    replay_client = reverb.Client(f"localhost:{replay_server.port}")
    counter = system.counter(False)
    trainer = system.trainer(replay_client, counter)
    evaluator_loop = system.evaluator(trainer, counter, trainer)
    return evaluator_loop


def get_trained_madqn_networks(
    substrate: str,
    network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
    checkpoint_dir: str,
) -> Dict[str, snt.Module]:
    """Obtains madqn networks trained on the substrate

    Args:
        substrate (str): substrate in which the networks were trained
        network_factory: creates the networks given the environment spec
        checkpoint_dir (str): checkpoint directory from which to restore network weights

    Returns:
        Dict[str, snt.Module]: trained networks
    """
    substrate_environment_factory = EnvironmentFactory(substrate=substrate)
    system = madqn.MADQN(
        environment_factory=substrate_environment_factory,
        network_factory=network_factory,
        exploration_scheduler_fn=LinearExplorationScheduler(
            epsilon_min=0.05, epsilon_decay=1e-4
        ),
        checkpoint_subpath=checkpoint_dir,
        shared_weights=False,
    )
    replay_server = reverb.Server(tables=system.replay())
    replay_client = reverb.Client(f"localhost:{replay_server.port}")
    trainer = system.trainer(replay_client, system.counter(False))
    return trainer._q_networks


def madqn_agent_network_setter(
    evaluator: Executor, trained_networks: Dict[str, snt.Module]
) -> None:
    """[summary]

    Args:
        evaluator (Executor): [description]
        trained_networks (Dict[str, snt.Module]): [description]
    """
    trained_network_keys = list(trained_networks.keys())
    for key in evaluator._q_networks.keys():
        idx = random.randint(0, len(trained_network_keys) - 1)
        evaluator._q_networks[key] = trained_networks[trained_network_keys[idx]]


def evaluate_on_scenarios(substrate, checkpoint_dir) -> None:
    """Tests the system on all the scenarios associated with the specified substrate"""
    scenarios = scenarios_for_substrate(substrate)

    # Networks.
    network_factory = lp_utils.partial_kwargs(make_default_madqn_networks)

    trained_networks = get_trained_madqn_networks(
        substrate, network_factory, checkpoint_dir
    )

    for scenario in scenarios:
        evaluate_on_scenario(scenario, network_factory, trained_networks)


def evaluate_on_scenario(
    scenario_name: str, network_factory, trained_networks: AgentNetworks
) -> None:
    """Tests the system on a given scenario"""

    # Scenario Environment.
    scenario_environment_factory = EnvironmentFactory(scenario=scenario_name)

    # Log every [log_every] seconds.
    log_every = 10

    def logger_factory(label, **kwargs):
        logger = logger_utils.make_logger(
            scenario_name,
            directory=FLAGS.logdir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=FLAGS.mava_id,
            time_delta=log_every,
        )
        return logger

    # Create madqn system for scenario
    scenario_system = madqn.MADQN(
        environment_factory=scenario_environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        exploration_scheduler_fn=LinearExplorationScheduler(
            epsilon_min=0.05, epsilon_decay=1e-4
        ),
        shared_weights=False,
    )

    # Evaluation loop
    evaluation_loop = ScenarioEvaluation(
        scenario_system,
        madqn_evaluation_loop_creator,
        madqn_agent_network_setter,
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

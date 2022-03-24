from mava.components.tf.modules.exploration.exploration_scheduling import LinearExplorationScheduler
from mava.systems.tf import value_decomposition
from mava.utils.environments.meltingpot_utils.env_utils import MeltingPotEnvironmentFactory
from mava.core import Executor
from mava.environment_loop import ParallelEnvironmentLoop
from mava.utils.environments.meltingpot_utils.evaluation_utils import (
    MAVASystem,
)
from acme import specs as acme_specs
import random
from typing import Any, Callable, Dict
import sonnet as snt


def qmix_evaluation_loop_creator(system: MAVASystem) -> ParallelEnvironmentLoop:
    """Creates an environment loop for the evaluation of a system

    Args:
        system ([MAVASystem]): the system to evaluate

    Returns:
        [ParallelEnvironmentLoop]: an environment loop for evaluation
    """
    evaluator_loop = system.evaluator(system.variable_server())
    return evaluator_loop



def get_trained_qmix_networks(
    substrate: str,
    network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
    checkpoint_dir: str,
) -> Dict[str, snt.Module]:
    """Obtains qmix networks trained on the substrate

    Args:
        substrate (str): substrate in which the networks were trained
        network_factory: creates the networks given the environment spec
        checkpoint_dir (str): checkpoint directory from which to restore network weights

    Returns:
        Dict[str, snt.Module]: trained networks
    """
    substrate_environment_factory = MeltingPotEnvironmentFactory(substrate=substrate)
    system = value_decomposition.ValueDecomposition(
        environment_factory=substrate_environment_factory,
        network_factory=network_factory,
        mixer="qmix",
        exploration_scheduler_fn=LinearExplorationScheduler(
            epsilon_min=0.05, epsilon_decay=1e-4
        ),
        checkpoint_subpath=checkpoint_dir,
        shared_weights=False,
    )
    networks = system.create_system()
    variables = system.variable_server().variables
    for net_type_key in networks:
        for net_key in networks[net_type_key]:
            for var_i in range(len(variables[f"{net_key}_{net_type_key}"])):
                networks[net_type_key][net_key].variables[var_i].assign(variables[f"{net_key}_{net_type_key}"][var_i])
    return networks  # type: ignore



def qmix_agent_network_setter(
    evaluator: Executor, trained_networks: Dict[str, Any]
) -> None:
    """Sets the networks for agents in the evaluator

    This is done by sampling from the trained networks

    Args:
        evaluator (Executor): [description]
        trained_networks (Dict[str, Any]): [description]
    """
    observation_networks = trained_networks["observations"]
    value_networks = trained_networks["values"]
    selectors = trained_networks["selectors"]
    
    # network keys
    trained_network_keys = list(trained_networks["observations"].keys())
    network_keys = evaluator._observation_networks.keys()
    
    for key in network_keys:
        # sample a trained network
        idx = random.randint(0, len(trained_network_keys) - 1)
        
        # observation networks
        evaluator._observation_networks[key]=observation_networks[trained_network_keys[idx]]
        
        # value networks
        evaluator._value_networks[key]=value_networks[trained_network_keys[idx]]
        
        # selectors
        evaluator._action_selectors[key]=selectors[trained_network_keys[idx]]


from typing import Dict, List, Optional

import sonnet as snt
import tensorflow as tf

from acme.tf import utils as tf_utils

from mava import specs as mava_specs
from mava.components.tf.networks.epsilon_greedy import EpsilonGreedy
from mava.utils.enums import ArchitectureType


class MeltingPotConvNet(snt.Module):
    def __init__(
        self,
        conv_out_channels: List[int] = [16, 32],
        conv_kernel_sizes: List[int] = [8, 4],
        conv_strides: List[int] = [8, 1],
        fc_layers: List[int] = [64, 64],
        name: Optional[str] = None,
    ) -> None:
        """Convolution network used to train agents in meltingpot

        Args:
            conv_out_channels (List[int], optional): output channels for the conv
            layers. Defaults to [16, 32].
            conv_kernel_sizes (List[int], optional): kernel size for the conv layers.
            Defaults to [8, 4].
            conv_strides (List[int], optional): strides for the conv layers.
            Defaults to [8, 1].
            fc_layers (List[int], optional): outputs nodes for the fc layers.
            Defaults to [64, 64].
            name (Optional[str], optional): name. Defaults to None.
        """
        super(MeltingPotConvNet, self).__init__(name=name)
        conv_layers = []
        fc_layers = []
        for output_channels, kernel_shape, stride in zip(
            conv_out_channels, conv_kernel_sizes, conv_strides
        ):
            conv_layer = snt.Conv2D(
                output_channels=output_channels,
                kernel_shape=kernel_shape,
                stride=stride,
            )
            conv_layers += [conv_layer]
            conv_layers += [tf.nn.relu]
        for nodes in fc_layers:
            fc_layer = snt.Linear(nodes)
            fc_layers += [fc_layer]
            fc_layers += [tf.nn.relu]
        all_layers = conv_layers + [snt.Flatten()] + fc_layers
        self._network = snt.Sequential(all_layers)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """
        return self._network(inputs)


def make_default_madqn_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    archecture_type: ArchitectureType = ArchitectureType.feedforward,
) -> Dict[str, snt.Module]:
    """Returns a network for madqn for melting pot envs

    Args:
        environment_spec (mava_specs.MAEnvironmentSpec): The environment specification
        agent_net_keys (Dict[str, str]): specifies networks for agent types
        archecture_type (ArchitectureType, optional): network architecture, recurrent or
            feedforward. Defaults to ArchitectureType.feedforward.

    Returns:
        Dict[str, snt.Module]: agent networks
    """
    specs = environment_spec.get_agent_specs()
    specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    q_networks = {}
    action_selectors = {}
    observation_networks = {}
    for key in specs.keys():
        num_dimensions = specs[key].actions.num_values
        if archecture_type == ArchitectureType.recurrent:
            network = snt.DeepRNN(
                [MeltingPotConvNet(), snt.LSTM(128), snt.Linear(num_dimensions)]
            )
        else:
            network = snt.Sequential([MeltingPotConvNet(), snt.Linear(num_dimensions)])
        q_networks[key] = network
        action_selectors[key] = EpsilonGreedy
        observation_networks[key] = tf_utils.to_sonnet_module(tf.identity)

    return {
        "values": q_networks,
        "action_selectors": action_selectors,
        "observations": observation_networks
    }



def make_default_qmix_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    archecture_type: ArchitectureType = ArchitectureType.feedforward,
) -> Dict[str, snt.Module]:
    """Returns a network for qmix for melting pot envs

    Args:
        environment_spec (mava_specs.MAEnvironmentSpec): The environment specification
        agent_net_keys (Dict[str, str]): specifies networks for agent types
        archecture_type (ArchitectureType, optional): network architecture, recurrent or
            feedforward. Defaults to ArchitectureType.feedforward.

    Returns:
        Dict[str, snt.Module]: agent networks
    """
    specs = environment_spec.get_agent_specs()
    specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    q_networks = {}
    action_selectors = {}
    observation_networks = {}
    for key in specs.keys():
        num_dimensions = specs[key].actions.num_values
        if archecture_type == ArchitectureType.recurrent:
            network = snt.DeepRNN(
                [MeltingPotConvNet(), snt.LSTM(128), snt.Linear(num_dimensions)]
            )
        else:
            network = snt.Sequential([MeltingPotConvNet(), snt.Linear(num_dimensions)])
        q_networks[key] = network
        action_selectors[key] = EpsilonGreedy
        observation_networks[key] = tf_utils.to_sonnet_module(tf.identity)

    return {
        "values": q_networks,
        "action_selectors": action_selectors,
        "observations": observation_networks
    }

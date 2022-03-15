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

"""Tests for MAPPO."""

import functools
from typing import Any, Dict, List, Union

import launchpad as lp
import numpy as np
import sonnet as snt
import tensorflow as tf

import mava
import reverb
from mava.adders.reverb.base import Trajectory
from mava.systems.tf import mappo
from mava.systems.tf.mappo.training import MAPPOTrainer
from mava.types import OLT
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils
from tests.utils.test_data import get_expected_parallel_timesteps_1

# from mava.adders.base import Trajectory


def get_mock_step_data():
    agents: List[Any] = ["agent_0", "agent_1", "agent_2"]

    default_action = {agent: 0.0 for agent in agents}
    reward_step1 = {"agent_0": 0.0, "agent_1": 0.0, "agent_2": 1.0}

    # (512, 11, 15)
    # (256, 3)

    batch_dim = 256
    sequence_dim = 11
    obs_dim = 15
    num_actions = 3

    discount = {agent: tf.ones([batch_dim, sequence_dim]) for agent in agents}
    olt_obs = OLT(
        observation=tf.ones([batch_dim, sequence_dim, obs_dim]),
        legal_actions=tf.ones([batch_dim, sequence_dim, num_actions]),
        terminal=tf.zeros([batch_dim, sequence_dim, 1]),
    )
    obs_first = {agent: olt_obs for agent in agents}

    data1 = Trajectory(
        observations=obs_first,
        actions=default_action,
        rewards=reward_step1,
        discounts=discount,
        start_of_episode=tf.ones([batch_dim, sequence_dim]),
        extras={},
    )

    return reverb.ReplaySample(
        info=reverb.SampleInfo(*[() for _ in reverb.SampleInfo.tf_dtypes()]),
        data=data1,
    )

    # return [
    #     reverb.ReplaySample(
    #         info=reverb.SampleInfo(*[() for _ in reverb.SampleInfo.tf_dtypes()]),
    #         data=data1,
    #     ),
    #     reverb.ReplaySample(
    #         info=reverb.SampleInfo(*[() for _ in reverb.SampleInfo.tf_dtypes()]),
    #         data=data1,
    #     ),
    # ]


class TestMAPPO:
    """Simple integration/smoke test for MAPPO."""

    def test_mappo_on_debugging_env(self) -> None:
        """Test feedforward mappo's trainer."""

        # trainer = MAPPOTrainer()

        agents: List[Any] = ["agent_0", "agent_1", "agent_2"]
        agent_types: List[str] = ["agent"]
        observation_network: Dict[str, snt.Module] = tf.identity

        # Make simple sonnet MLP
        policy_network = snt.nets.MLP([2])

        # Make simple sonnet MLP
        critic_network = snt.nets.MLP([2])

        # Pass observation through simple MLP to initialize network variables
        mock_obs = tf.ones([100, 1])
        test_out_policy = policy_network(mock_obs)
        test_out_critic = critic_network(mock_obs)

        policy_networks = {}
        critic_networks = {}
        observation_networks = {}
        for a in agent_types:
            policy_networks[a] = policy_network
            critic_networks[a] = critic_network
            observation_networks[a] = observation_network

        observation = get_mock_step_data()
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor(
            observation
        ).as_numpy_iterator()
        # dataset = iter(observation)
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]] = snt.optimizers.Adam(
            learning_rate=5e-4
        )
        agent_net_keys = {
            "agent_0": "agent",
            "agent_1": "agent",
            "agent_2": "agent",
        }
        checkpoint_minute_interval: int = 100

        trainer = MAPPOTrainer(
            agents=agents,
            agent_types=agent_types,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            dataset=dataset,
            optimizer=optimizer,
            agent_net_keys=agent_net_keys,
            checkpoint_minute_interval=checkpoint_minute_interval,
        )

        for _ in range(2):
            trainer.step()

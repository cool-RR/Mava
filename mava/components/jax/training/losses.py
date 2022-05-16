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

"""Trainer components for calculating losses."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import rlax
from haiku._src.basic import merge_leading_dims

from mava.components.jax.training.base import Loss
from mava.core_jax import SystemTrainer


@dataclass
class MAPGTrustRegionClippingLossConfig:
    clipping_epsilon: float = 0.2
    clip_value: bool = True
    entropy_cost: float = 0.01
    value_cost: float = 0.5


class MAPGWithTrustRegionClippingLoss(Loss):
    def __init__(
        self,
        config: MAPGTrustRegionClippingLossConfig = MAPGTrustRegionClippingLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def loss_grad_fn(
            params: Any,
            observations: Any,
            actions: Dict[str, jnp.ndarray],
            behaviour_log_probs: Dict[str, jnp.ndarray],
            target_values: Dict[str, jnp.ndarray],
            advantages: Dict[str, jnp.ndarray],
            behavior_values: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios."""

            grads = {}
            loss_info = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks["networks"][agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def loss_fn(
                    params: Any,
                    observations: Any,
                    actions: jnp.ndarray,
                    behaviour_log_probs: jnp.ndarray,
                    target_values: jnp.ndarray,
                    advantages: jnp.ndarray,
                    behavior_values: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                    distribution_params, values = network.network.apply(
                        params, observations
                    )
                    log_probs = network.log_prob(distribution_params, actions)
                    entropy = network.entropy(distribution_params)
                    # Compute importance sampling weights:
                    # current policy / behavior policy.
                    rhos = jnp.exp(log_probs - behaviour_log_probs)
                    clipping_epsilon = self.config.clipping_epsilon

                    policy_loss = rlax.clipped_surrogate_pg_loss(
                        rhos, advantages, clipping_epsilon
                    )

                    # Value function loss. Exclude the bootstrap value
                    unclipped_value_error = target_values - values
                    unclipped_value_loss = unclipped_value_error**2

                    if self.config.clip_value:
                        # Clip values to reduce variablility during critic training.
                        clipped_values = behavior_values + jnp.clip(
                            values - behavior_values,
                            -clipping_epsilon,
                            clipping_epsilon,
                        )
                        clipped_value_error = target_values - clipped_values
                        clipped_value_loss = clipped_value_error**2
                        value_loss = jnp.mean(
                            jnp.fmax(unclipped_value_loss, clipped_value_loss)
                        )
                    else:
                        value_loss = jnp.mean(unclipped_value_loss)

                    # Entropy regulariser.
                    entropy_loss = -jnp.mean(entropy)

                    total_loss = (
                        policy_loss
                        + value_loss * self.config.value_cost
                        + entropy_loss * self.config.entropy_cost
                    )

                    loss_info = {
                        "loss_total": total_loss,
                        "loss_policy": policy_loss,
                        "loss_value": value_loss,
                        "loss_entropy": entropy_loss,
                    }

                    return total_loss, loss_info

                grads[agent_key], loss_info[agent_key] = jax.grad(
                    loss_fn, has_aux=True
                )(
                    params[agent_net_key],
                    observations[agent_key].observation,
                    actions[agent_key],
                    behaviour_log_probs[agent_key],
                    target_values[agent_key],
                    advantages[agent_key],
                    behavior_values[agent_key],
                )
            return grads, loss_info

        # Save the gradient funciton.
        trainer.store.grad_fn = loss_grad_fn

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return MAPGTrustRegionClippingLossConfig


@dataclass
class MAMCTSLossConfig:
    L2_regularisation_coeff: float = 0.0001
    value_cost: float = 1.0


class MAMCTSLoss(Loss):
    def __init__(
        self,
        config: MAMCTSLossConfig = MAMCTSLossConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_training_loss_fns(self, trainer: SystemTrainer) -> None:
        """_summary_"""

        def loss_grad_fn(
            params: Any,
            observations: Any,
            search_policies: Dict[str, jnp.ndarray],
            target_values: Dict[str, jnp.ndarray],
            recieved_messages,
            search_trees,
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """Surrogate loss using clipped probability ratios."""

            grads = {}
            loss_info = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks["networks"][agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def loss_fn(
                    params: Any,
                    observations: jnp.ndarray,
                    search_policies: jnp.ndarray,
                    target_values: jnp.ndarray,
                    recieved_messages: jnp.ndarray,
                    search_trees: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

                    (
                        merged_agent_obs,
                        merged_agent_trees,
                        merged_agent_recieved_messages,
                    ) = ([], [], [])
                    for other_agent_key in trainer.store.trainer_agents:
                        (
                            merged_obs,
                            merged_trees,
                            merged_recieved_messages,
                        ) = jax.tree_map(
                            lambda x: merge_leading_dims(x, 2),
                            (
                                observations[other_agent_key].observation,
                                search_trees[other_agent_key],
                                recieved_messages[other_agent_key],
                            ),
                        )
                        merged_agent_obs.append(merged_obs)
                        merged_agent_trees.append(merged_trees)
                        merged_agent_recieved_messages.append(merged_recieved_messages)

                    merged_obs = jnp.concatenate(merged_agent_obs, 0)
                    merged_trees = jnp.concatenate(merged_agent_trees, 0)
                    merged_recieved_messages = jnp.concatenate(
                        merged_agent_recieved_messages, 0
                    )

                    (_, _), message = network.network.apply(
                        params, merged_obs, merged_trees, merged_recieved_messages
                    )

                    agent_messages = message.reshape(
                        -1, len(trainer.store.trainer_agents) * 20
                    )

                    agent_messages = agent_messages.reshape(
                        *search_trees[agent_key].shape[0:2], *agent_messages.shape[-1:]
                    )
                    agent_messages = jax.tree_map(lambda x: x[:, :-1], agent_messages)
                    agent_messages = jax.tree_map(
                        lambda x: merge_leading_dims(x, 2), agent_messages
                    )

                    (
                        merged_obs,
                        merged_trees,
                        search_policies,
                        target_values,
                    ) = jax.tree_map(
                        lambda x: merge_leading_dims(x[:, 1:], 2),
                        (
                            observations[agent_key].observation,
                            search_trees[agent_key],
                            search_policies,
                            target_values,
                        ),
                    )

                    (logits, values), _ = network.network.apply(
                        params, merged_obs, merged_trees, agent_messages
                    )

                    policy_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            search_policies, logits.logits
                        )
                    )

                    value_loss = jnp.mean(rlax.l2_loss(values, target_values))

                    # Entropy regulariser.
                    l2_regularisation = sum(
                        jnp.sum(jnp.square(parameter))
                        for parameter in jax.tree_leaves(params)
                    )

                    total_loss = (
                        policy_loss
                        + value_loss * self.config.value_cost
                        + l2_regularisation * self.config.L2_regularisation_coeff
                    )

                    loss_info = {
                        "loss_total": total_loss,
                        "loss_policy": policy_loss,
                        "loss_value": value_loss,
                        "loss_regularisation_term": l2_regularisation,
                    }

                    return total_loss, loss_info

                grads[agent_key], loss_info[agent_key] = jax.grad(
                    loss_fn, has_aux=True
                )(
                    params[agent_net_key],
                    observations,
                    search_policies[agent_key],
                    target_values[agent_key],
                    recieved_messages,
                    search_trees,
                )
            return grads, loss_info

        # Save the gradient funciton.
        trainer.store.grad_fn = loss_grad_fn

    @staticmethod
    def config_class() -> Callable:
        return MAMCTSLossConfig

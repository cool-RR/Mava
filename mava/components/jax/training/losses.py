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

import haiku as hk
import jax
import jax.numpy as jnp
import rlax
from haiku._src.basic import merge_leading_dims

from mava.components.jax.training.base import Loss
from mava.core_jax import SystemTrainer
from mava.systems.jax.mamcts.learned_model_utils import (
    scalar_to_two_hot,
    scale_gradient,
    value_transform,
)


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
    """MAMCTS Loss - essentially a decentralised AlphaZero loss"""

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
            rewards: Dict[str, jnp.ndarray],
            actions: Dict[str, jnp.ndarray],
            observation_history: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """TODO add description"""

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
                    search_policies: jnp.ndarray,
                    target_values: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

                    logits, value_logits = network.forward_fn(params, observations)

                    # Transform the target values into logits
                    target_values = value_transform(target_values)
                    target_values_logits = scalar_to_two_hot(
                        target_values, network._num_bins
                    )
                    target_values_logits = jax.lax.stop_gradient(target_values_logits)

                    # Compute the policy loss
                    policy_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            search_policies, logits
                        )
                    )

                    # Compute the value loss
                    value_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            target_values_logits, value_logits
                        )
                    )

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
                    observations[agent_key].observation,
                    search_policies[agent_key],
                    target_values[agent_key],
                )
            return grads, loss_info

        # Save the gradient funciton.
        trainer.store.grad_fn = loss_grad_fn

    @staticmethod
    def config_class() -> Callable:
        return MAMCTSLossConfig


class MAMCTSLearnedModelLoss(Loss):
    """MAMCTS Loss - essentially a decentralised AlphaZero loss"""

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
            search_policies: Dict[str, jnp.ndarray],
            target_values: Dict[str, jnp.ndarray],
            rewards: Dict[str, jnp.ndarray],
            actions: Dict[str, jnp.ndarray],
            observation_history: Dict[str, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            """TODO add description"""

            grads = {}
            loss_info = {}
            for agent_key in trainer.store.trainer_agents:
                agent_net_key = trainer.store.trainer_agent_net_keys[agent_key]
                network = trainer.store.networks["networks"][agent_net_key]

                # Note (dries): This is placed here to set the networks correctly in
                # the case of non-shared weights.
                def loss_fn(
                    params: Any,
                    search_policies: jnp.ndarray,
                    target_values: jnp.ndarray,
                    rewards: jnp.ndarray,
                    actions: jnp.ndarray,
                    observation_history: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:

                    # Get batch initial root embeddings
                    initial_observation_history = observation_history[:, 0]

                    root_embeddings = network.representation_network.network.apply(
                        params["representation"], initial_observation_history
                    )

                    def dynamics_step(action, prev_state) -> Tuple[Any, Any]:
                        """Run one step of the RNN.
                        Args:
                        inputs: An arbitrarily nested structure.
                        prev_state: Previous core state.
                        Returns:
                        A tuple with two elements ``output, next_state``. ``output`` is an
                        arbitrarily nested structure. ``next_state`` is the next core state, this
                        must be the same shape as ``prev_state``."""

                        (
                            new_embedding,
                            reward_logits,
                        ) = network.dynamics_network.network.apply(
                            params["dynamics"], prev_state, action
                        )

                        new_embedding = scale_gradient(new_embedding, 0.5)
                        return reward_logits, new_embedding

                    # unroll and get the predicted reward logits and embeddings
                    predicted_rewards_logits, predicted_embeddings = hk.dynamic_unroll(
                        dynamics_step,
                        actions,
                        root_embeddings,
                        time_major=False,
                        return_all_states=True,
                    )

                    # Transform the rewards into logits
                    rewards = value_transform(rewards)
                    rewards_logits = scalar_to_two_hot(rewards, network._num_bins)

                    # Transform the target values into logits
                    target_values = value_transform(target_values)
                    target_values_logits = scalar_to_two_hot(
                        target_values, network._num_bins
                    )
                    target_values_logits = jax.lax.stop_gradient(target_values_logits)

                    # Add the initial root embedding to the sequence of generated embeddings
                    predicted_embeddings = jnp.concatenate(
                        [
                            jnp.expand_dims(root_embeddings, 1),
                            predicted_embeddings[:, 0:-1],
                        ],
                        axis=1,
                    )

                    # Get the policy and value logits for each of the generated embeddings
                    logits, value_logits = network.prediction_network.network.apply(
                        params["prediction"],
                        merge_leading_dims(predicted_embeddings, 2),
                    )

                    # Compute the policy loss
                    policy_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            merge_leading_dims(search_policies, 2), logits
                        )
                    )

                    # Compute the value loss
                    value_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            merge_leading_dims(target_values_logits, 2), value_logits
                        )
                    )

                    # Compute the reward loss
                    reward_loss = jnp.mean(
                        jax.vmap(rlax.categorical_cross_entropy, in_axes=(0, 0))(
                            merge_leading_dims(rewards_logits, 2),
                            merge_leading_dims(predicted_rewards_logits, 2),
                        )
                    )

                    # Entropy regulariser.
                    l2_regularisation = sum(
                        jnp.sum(jnp.square(parameter))
                        for parameter in jax.tree_leaves(params)
                    )

                    # Scale the gradients by 1/N where N is sequence length
                    sequence_length = actions.shape[-1]
                    policy_loss = scale_gradient(policy_loss, 1 / sequence_length)
                    value_loss = scale_gradient(value_loss, 1 / sequence_length)
                    reward_loss = scale_gradient(reward_loss, 1 / sequence_length)

                    total_loss = (
                        policy_loss
                        + self.config.value_cost * value_loss
                        + reward_loss
                        + self.config.L2_regularisation_coeff * l2_regularisation
                    )

                    loss_info = {
                        "loss_total": total_loss,
                        "loss_policy": policy_loss,
                        "loss_value": value_loss,
                        "loss_reward": reward_loss,
                        "loss_regularisation_term": l2_regularisation,
                    }

                    return total_loss, loss_info

                grads[agent_key], loss_info[agent_key] = jax.grad(
                    loss_fn, has_aux=True
                )(
                    params[agent_net_key],
                    search_policies[agent_key],
                    target_values[agent_key],
                    rewards[agent_key],
                    actions[agent_key],
                    observation_history[agent_key],
                )
            return grads, loss_info

        # Save the gradient funciton.
        trainer.store.grad_fn = loss_grad_fn

    @staticmethod
    def config_class() -> Callable:
        return MAMCTSLossConfig

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "loss_fn"

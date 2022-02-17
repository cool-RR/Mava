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

import copy
import math

import tensorflow as tf
import trfl
from acme.tf import losses

from mava.components.tf.networks import DiscreteValuedDistribution
from mava.utils.training_utils import check_rank, combine_dim


def recurrent_n_step_critic_loss(
    q_values: tf.Tensor,
    target_q_values: tf.Tensor,
    rewards: tf.Tensor,
    bootstrap_n: int,
    discount: float,
    end_of_episode: tf.Tensor,
) -> tf.Tensor:

    # d: discount * done
    # bootstrap_n=1 is the normal return of Q_t-1 = R_t-1 + d * Q_t
    # bootstrap_n=seq_len is the Q_t_1 = discounted sum of rewards return

    seq_len = len(rewards[0])
    assert 0 < bootstrap_n <= seq_len

    if not isinstance(q_values, DiscreteValuedDistribution):
        # TODO (dries): Implement this test for MAD4PG as well.
        check_rank([q_values, target_q_values, rewards], [2, 2, 2])

    # The last values that rolled over do not matter because a
    # mask is applied to it.
    if not isinstance(q_values, DiscreteValuedDistribution):
        # Construct arguments to compute bootstrap target.
        q_tm1, _ = combine_dim(q_values)
        q_t, _ = combine_dim(tf.roll(target_q_values, shift=-bootstrap_n, axis=1))
    else:
        q_tm1 = q_values
        q_t = copy.copy(q_values)

        # Roll the logits inside the tfp distribution
        # Question (dries): Is there a more elegant way to do this that working with
        # _logits directly?
        last_dim_shape = (q_t._logits.shape[-1],)
        reshaped_logits = tf.reshape(q_t._logits, (rewards.shape + last_dim_shape))
        rolled_logits = tf.roll(reshaped_logits, shift=-bootstrap_n, axis=1)
        q_t._logits, _ = combine_dim(rolled_logits)

    # Pad the rewards so that rewards at the end can also be calculated.
    r_shape = rewards.shape
    zeros_mask = tf.zeros(shape=r_shape[:-1] + (r_shape[-1] - bootstrap_n - 1,))
    padded_rewards = tf.concat([rewards, zeros_mask], axis=1)
    n_step_rewards = rewards
    for i in range(1, bootstrap_n):
        n_step_rewards += padded_rewards[:, i : i + seq_len] * math.pow(discount, i)
    n_step_rewards, _ = combine_dim(n_step_rewards)

    # Create the end of episode mask.
    ones_mask = tf.ones(shape=r_shape[:-1] + (r_shape[-1] - bootstrap_n,))
    zeros_mask = tf.zeros(shape=r_shape[:-1] + (bootstrap_n,))
    end_of_episode_mask, _ = combine_dim(tf.concat([ones_mask, zeros_mask], axis=1))

    # Role episode done masking
    done_masking, _ = combine_dim(tf.roll(end_of_episode, shift=-bootstrap_n, axis=1))

    flat_mask = end_of_episode_mask * done_masking * math.pow(discount, bootstrap_n)
    if isinstance(q_values, DiscreteValuedDistribution):
        critic_loss = losses.categorical(
            q_tm1=q_tm1,
            r_t=n_step_rewards,
            d_t=flat_mask,
            q_t=q_t,
        )
    else:
        critic_loss = trfl.td_learning(
            v_tm1=q_tm1,
            r_t=n_step_rewards,
            pcont_t=flat_mask,
            v_t=q_t,
        ).loss

    return critic_loss

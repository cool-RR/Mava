from typing import Callable

import jax
import jax.numpy as jnp
import mctx
from acme.jax import utils
from mctx._src.policies import _mask_invalid_actions

from mava.utils.tree_utils import add_batch_dim_tree, remove_batch_dim_tree, stack_trees
from mava.wrappers.env_wrappers import EnvironmentModelWrapper


def generic_root_fn():
    def root_fn(forward_fn, params, rng_key, env_state, observation):
        prior_logits, values, _ = forward_fn(
            observations=observation, params=params, key=rng_key
        )

        return mctx.RootFnOutput(
            prior_logits=prior_logits.logits,
            value=values,
            embedding=add_batch_dim_tree(env_state),
        )

    return root_fn


def default_action_recurrent_fn(default_action, discount_gamma=0.99) -> Callable:
    def recurrent_fn(
        environment_model: EnvironmentModelWrapper,
        forward_fn,
        params,
        rng_key,
        action,
        env_state,
        agent_info,
    ) -> mctx.RecurrentFnOutput:
        agent_list = environment_model.get_possible_agents()

        actions = {agent_id: default_action for agent_id in agent_list}

        actions[agent_info] = jnp.squeeze(action)

        env_state = remove_batch_dim_tree(env_state)

        next_state, timestep, _ = environment_model.step(env_state, actions)

        observation = environment_model.get_observation(next_state, agent_info)

        prior_logits, values = forward_fn(
            observations=utils.add_batch_dim(observation), params=params, key=rng_key
        )

        agent_mask = utils.add_batch_dim(
            environment_model.get_agent_mask(next_state, agent_info)
        )

        prior_logits = _mask_invalid_actions(prior_logits.logits, agent_mask)

        reward = timestep.reward[agent_info].reshape(
            1,
        )

        discount = (
            timestep.discount[agent_info].reshape(
                1,
            )
            * discount_gamma
        )

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=values,
            ),
            add_batch_dim_tree(next_state),
        )

    return recurrent_fn


def random_action_recurrent_fn(discount_gamma=0.99) -> Callable:
    def recurrent_fn(
        environment_model: EnvironmentModelWrapper,
        forward_fn,
        params,
        rng_key,
        action,
        env_state,
        agent_info,
    ) -> mctx.RecurrentFnOutput:
        agent_list = environment_model.get_possible_agents()

        rng_key, *agent_action_keys = jax.random.split(rng_key, len(agent_list))

        actions = {
            agent_id: jax.random.randint(
                agent_rng_key,
                (),
                minval=0,
                maxval=environment_model.action_spec()[agent_info].num_values,
            )
            for agent_rng_key, agent_id in zip(agent_action_keys, agent_list)
        }

        actions[agent_info] = jnp.squeeze(action)

        env_state = remove_batch_dim_tree(env_state)

        next_state, timestep, _ = environment_model.step(env_state, actions)

        observation = environment_model.get_observation(next_state, agent_info)

        prior_logits, values = forward_fn(
            observations=utils.add_batch_dim(observation), params=params, key=rng_key
        )

        agent_mask = utils.add_batch_dim(
            environment_model.get_agent_mask(next_state, agent_info)
        )

        prior_logits = _mask_invalid_actions(prior_logits.logits, agent_mask)

        reward = timestep.reward[agent_info].reshape(
            1,
        )

        discount = (
            timestep.discount[agent_info].reshape(
                1,
            )
            * discount_gamma
        )

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=values,
            ),
            add_batch_dim_tree(next_state),
        )

    return recurrent_fn


def greedy_policy_recurrent_fn(discount_gamma=0.99) -> Callable:
    def recurrent_fn(
        environment_model: EnvironmentModelWrapper,
        forward_fn,
        params,
        rng_key,
        action,
        env_state,
        agent_info,
    ) -> mctx.RecurrentFnOutput:
        agent_list = environment_model.get_possible_agents()

        stacked_agents = stack_trees(agent_list)

        env_state = remove_batch_dim_tree(env_state)

        prev_observations = jax.vmap(
            environment_model.get_observation, in_axes=(None, 0)
        )(env_state, stacked_agents)

        _, _, prev_prior_logits = forward_fn(
            observations=prev_observations, params=params, key=rng_key
        )

        other_agent_masks = jax.vmap(
            environment_model.get_agent_mask, in_axes=(None, 0)
        )(env_state, stacked_agents)

        prev_prior_logits = jax.vmap(_mask_invalid_actions, in_axes=(0, 0))(
            prev_prior_logits.logits, other_agent_masks
        )

        agent_actions = jnp.argmax(prev_prior_logits, -1)

        actions = {agent_id: agent_actions[agent_id.id] for agent_id in agent_list}

        actions[agent_info] = jnp.squeeze(action)

        next_state, timestep, _ = environment_model.step(env_state, actions)

        observation = environment_model.get_observation(next_state, agent_info)

        prior_logits, values, _ = forward_fn(
            observations=utils.add_batch_dim(observation), params=params, key=rng_key
        )

        agent_mask = utils.add_batch_dim(
            environment_model.get_agent_mask(next_state, agent_info)
        )

        prior_logits = _mask_invalid_actions(prior_logits.logits, agent_mask)

        reward = timestep.reward[agent_info].reshape(
            1,
        )

        discount = (
            timestep.discount[agent_info].reshape(
                1,
            )
            * discount_gamma
        )

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=values,
            ),
            add_batch_dim_tree(next_state),
        )

    return recurrent_fn

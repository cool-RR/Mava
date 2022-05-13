import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import acme.jax.utils as utils
import chex
import jax.numpy as jnp
import mctx
import numpy as np
from haiku import Params
from jax import jit

from mava.utils.id_utils import EntityId
from mava.utils.tree_utils import (
    add_batch_dim_tree,
    apply_fun_tree,
    remove_batch_dim_tree,
)
from mava.wrappers.env_wrappers import EnvironmentModelWrapper

RecurrentState = Any
RootFn = Callable[[Params, chex.PRNGKey, Any], mctx.RootFnOutput]
RecurrentState = Any
RecurrentFn = Callable[
    [Params, chex.PRNGKey, chex.Array, RecurrentState],
    Tuple[mctx.RecurrentFnOutput, RecurrentState],
]
MaxDepth = Optional[int]
SearchOutput = mctx.PolicyOutput[Union[mctx.GumbelMuZeroExtraData, None]]
TreeSearch = Callable[
    [Params, chex.PRNGKey, mctx.RootFnOutput, RecurrentFn, int, MaxDepth], SearchOutput
]


class MCTS:
    """TODO: Add description here."""

    def __init__(self, config) -> None:
        """TODO: Add description here."""
        self.config = config

    def get_action(
        self,
        forward_fn,
        params,
        rng_key,
        env_state,
        observation,
        agent_info,
    ):
        """TODO: Add description here."""
        # agent_info = EntityId.from_string(agent_info)
        search_out = self.search(
            forward_fn,
            params,
            rng_key,
            env_state,
            observation,
            agent_info,
        )
        action = jnp.squeeze(search_out.action.astype(jnp.int32))
        search_policy = jnp.squeeze(search_out.action_weights)

        return (action, {"search_policies": search_policy})

    @functools.partial(jit, static_argnames=["self", "forward_fn", "agent_info"])
    def search(
        self,
        forward_fn,
        params,
        rng_key,
        env_state,
        observation,
        agent_info,
    ):
        """TODO: Add description here."""

        root = self.config.root_fn(forward_fn, params, rng_key, env_state, observation)

        def recurrent_fn(params, rng_key, action, embedding):

            return self.config.recurrent_fn(
                self.config.environment_model,
                forward_fn,
                params,
                rng_key,
                action,
                embedding,
                agent_info,
            )

        root_invalid_actions = utils.add_batch_dim(
            self.config.environment_model.get_agent_mask(env_state, agent_info)
        )

        search_output = self.config.search(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.config.num_simulations,
            invalid_actions=root_invalid_actions,
            max_depth=self.config.max_depth,
        )

        return search_output

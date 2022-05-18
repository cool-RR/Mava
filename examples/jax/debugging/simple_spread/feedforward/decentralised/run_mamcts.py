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

"""Example running MAMCTS on debug MPE environments."""
import functools
from datetime import datetime
from typing import Any

import haiku as hk
import mctx
import optax
from absl import app, flags
from acme.jax import utils
from acme.jax.networks.atari import DeepAtariTorso
from marlin.mava_exps.environments.debug_env.debug_grid_env_wrapper import (
    DebugEnvWrapper,
)
from mctx import RecurrentFnOutput, RootFnOutput

from mava.systems.jax import mamcts
from mava.systems.jax.mamcts.mcts_utils import (
    default_action_recurrent_fn,
    generic_root_fn,
    greedy_policy_recurrent_fn,
    random_action_recurrent_fn,
)
from mava.utils.debugging.environments.jax.debug_env.new_debug_env import DebugEnv
from mava.utils.loggers import logger_utils
from mava.wrappers.environment_loop_wrappers import (
    JAXDetailedEpisodeStatistics,
    JAXDetailedPerAgentStatistics,
    JAXMonitorEnvironmentLoop,
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "debug_env",
    "Debugging environment name (str).",
)
flags.DEFINE_string(
    "action_space",
    "discrete",
    "Environment action space type (str).",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


def make_environment(rows=8, cols=8, evaluation: bool = None, num_agents: int = 3):

    return DebugEnvWrapper(
        DebugEnv(
            rows,
            cols,
            num_agents,
            reward_for_connection=1.0,
            reward_for_blocked=-1.0,
            reward_per_timestep=-1.0 / (rows + cols),
        )
    )


def network_factory(
    policy_layer_sizes=(64,), critic_layer_sizes=(64,), *args, **kwargs
):
    obs_net_forward = lambda x: hk.Sequential([hk.Embed(128, 8), DeepAtariTorso()])(
        x.astype(int)
    )
    return mamcts.make_default_networks(
        policy_layer_sizes=policy_layer_sizes,
        critic_layer_sizes=critic_layer_sizes,
        observation_network=obs_net_forward,
        *args,
        **kwargs,
    )


def main(_: Any) -> None:
    """Run main script

    Args:
        _ : _
    """
    # Environment.
    environment_factory = functools.partial(make_environment)

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_subpath = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # Log every [log_every] seconds.
    log_every = 5
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=FLAGS.mava_id,
        time_delta=log_every,
    )

    # Optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-3)
    )

    # Create the system.
    system = mamcts.MAMCTSSystem()

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        run_evaluator=True,
        sample_batch_size=16,
        # num_minibatches=8,
        num_epochs=2,
        num_executors=1,
        multi_process=True,
        root_fn=generic_root_fn(),
        recurrent_fn=default_action_recurrent_fn(0, discount_gamma=1.0),
        search=mctx.gumbel_muzero_policy,
        environment_model=environment_factory(),
        num_simulations=15,
        evaluator_num_simulations=50,
        evaluator_other_search_params=lambda: {"gumbel_scale": 0.0},
        rng_seed=0,
        n_step=10,
        discount=0.997,
        executor_stats_wrapper_class=JAXDetailedEpisodeStatistics,
        # evaluator_stats_wrapper_class=JAXMonitorEnvironmentLoop,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)

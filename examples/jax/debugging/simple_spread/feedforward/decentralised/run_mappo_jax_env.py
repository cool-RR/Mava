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

"""Example running MAPPO on debug MPE environments."""
import functools
from datetime import datetime
from typing import Any

import jax
import haiku as hk
from acme.jax.networks.atari import DeepAtariTorso
import optax
from absl import app, flags

from mava.components.jax.building.environments import JAXParallelExecutorEnvironmentLoop
from mava.systems.jax import mappo
from mava.utils.debugging.environments.jax.debug_env.new_debug_env import DebugEnv
from mava.utils.loggers import logger_utils
from mava.wrappers.JaxDebugEnvWrapper import DebugEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "simple_spread",
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


def make_environment(rows=6, cols=6, evaluation: bool = None, num_agents: int = 1):

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
    policy_layer_sizes=(128,), critic_layer_sizes=(512, 512), *args, **kwargs
):
    obs_net_forward = lambda x: hk.Sequential([hk.Embed(128, 8), DeepAtariTorso()])(
        x.astype(int)
    )
    return mappo.make_default_networks(
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

    environment_factory = functools.partial(
        make_environment,
    )

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
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Create the system.
    system = mappo.MAPPOSystem()

    system.update(JAXParallelExecutorEnvironmentLoop)

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=mappo.make_default_networks,
        logger_factory=logger_factory,
        checkpoint_subpath=checkpoint_subpath,
        optimizer=optimizer,
        run_evaluator=True,
        sample_batch_size=256,
        num_minibatches=8,
        num_epochs=4,
        num_executors=6,
        multi_process=True,
        learning_rate=0.001,
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)

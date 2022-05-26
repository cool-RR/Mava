import time
from typing import Any, Callable, Dict, Optional, Tuple

import dm_env
import numpy as np
from acme.utils import counting, loggers
from chex import dataclass

import mava
from mava.components.jax.building import ParallelExecutorEnvironmentLoop
from mava.components.jax.building.environments import ExecutorEnvironmentLoopConfig
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.jax.hrl.executor import HrlExecutor
from mava.systems.jax.hrl.hrl_wrapper import HrlEnvironmentWrapper
from mava.utils.training_utils import check_count_condition
from mava.utils.wrapper_utils import generate_zeros_from_spec


@dataclass
class HrlExecutorEnvironmentLoopConfig(ExecutorEnvironmentLoopConfig):
    hrl_interval: int = 5


class HrlParallelExecutorEnvironmentLoop(ParallelExecutorEnvironmentLoop):
    def __init__(
        self,
        config: HrlExecutorEnvironmentLoopConfig = HrlExecutorEnvironmentLoopConfig(),
    ):
        """[summary]"""
        self.config = config

    def on_building_executor_environment_loop(self, builder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        executor_environment_loop = HrlParallelEnvironmentLoop(
            environment=builder.store.executor_environment,
            executor=builder.store.executor,
            logger=builder.store.executor_logger,
            should_update=self.config.should_update,
            hrl_interval=self.config.hrl_interval,
        )

        del builder.store.executor_logger

        builder.store.system_executor = executor_environment_loop

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return HrlExecutorEnvironmentLoopConfig


class HrlParallelEnvironmentLoop(ParallelEnvironmentLoop):
    """A parallel MARL environment loop.

    This takes `Environment` and `Executor` instances and coordinates their
    interaction. Executors are updated if `should_update=True`. This can be used as:
        loop = EnvironmentLoop(environment, executor)
        loop.run(num_episodes)
    A `Counter` instance can optionally be given in order to maintain counts
    between different Mava components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.
    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger from acme. A string `label` can be passed
    to easily change the label associated with the default logger; this is ignored
    if a `Logger` instance is given.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        executor: HrlExecutor,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        should_update: bool = True,
        label: str = "parallel_environment_loop",
        hrl_interval: int = 5,
    ):
        """Parallel environment loop init

        Args:
            environment: an environment
            executor: a Mava executor
            counter: an optional counter. Defaults to None.
            logger: an optional counter. Defaults to None.
            should_update: should update. Defaults to True.
            label: optional label. Defaults to "sequential_environment_loop".
        """
        assert isinstance(executor, HrlExecutor)
        assert isinstance(environment, HrlEnvironmentWrapper)

        super().__init__(environment, executor, counter, logger, should_update, label)
        self.hrl_interval = hrl_interval

    @staticmethod
    def get_extras(timestep):
        if type(timestep) == tuple:
            timestep, env_extras = timestep
        else:
            env_extras = {}

        return timestep, env_extras

    def _get_actions(self, timestep: dm_env.TimeStep) -> Any:
        actions = self._executor.select_ll_actions(timestep.observation)
        if type(actions) == tuple:
            # Return other action information
            # e.g. the policy information.
            return actions[0]

        return actions

    def observe_first(self, timestep, env_extras):
        self._executor.hl_observe_first(timestep, extras=env_extras)
        hl_actions, _ = self._executor.select_hl_actions(timestep.observation)
        ll_timestep = self._environment.ll_timestep(timestep, hl_actions)
        # env_extras["hl_actions"] = hl_actions
        self._executor.ll_observe_first(ll_timestep, extras={})

        return ll_timestep, hl_actions

    def observe(self, prev_hl_actions, prev_ll_actions, timestep, env_extras):
        self._executor.hl_observe(prev_hl_actions, timestep, env_extras)
        hl_actions, _ = self._executor.select_hl_actions(timestep.observation)
        ll_timestep = self._environment.ll_timestep(timestep, hl_actions)
        # env_extras["hl_actions"] = hl_actions
        self._executor.ll_observe(prev_ll_actions, ll_timestep, {})

        return ll_timestep, hl_actions

    def run_episode(self) -> loggers.LoggingData:
        """Run one episode.

        Each episode is a loop which interacts first with the environment to get a
        dictionary of observations and then give those observations to the executor
        in order to retrieve an action for each agent in the system.

        Returns:
            An instance of `loggers.LoggingData`.
        """

        # Reset any counts and start the environment.
        start_time = time.time()
        episode_steps = 0

        timestep = self._environment.reset()
        timestep, env_extras = self.get_extras(timestep)
        # Make the first observation.
        ll_timestep, hl_actions = self.observe_first(timestep, env_extras)

        # For evaluation, this keeps track of the total undiscounted reward
        # for each agent accumulated during the episode.
        rewards: Dict[str, float] = {}
        episode_returns: Dict[str, float] = {}
        for agent, spec in self._environment.reward_spec()["hl"].items():
            rewards.update({agent: generate_zeros_from_spec(spec)})
            episode_returns.update({agent: generate_zeros_from_spec(spec)})

        action_counts = {i: 0 for i in range(5)}
        # Run an episode.
        while not timestep.last():
            env_actions = self._get_actions(ll_timestep)
            # env_actions = hl_actions

            timestep = self._environment.step(env_actions)
            timestep, env_extras = self.get_extras(timestep)
            rewards = timestep.reward

            action_counts[env_actions['id-0-type-0'].item()] += 1

            # Have the agent observe the timestep and let the actor update itself.
            ll_timestep, hl_actions = self.observe(
                hl_actions, env_actions, timestep, env_extras
            )

            if self._should_update:
                self._executor.update()

            # Book-keeping.
            episode_steps += 1  # TODO ll and hl steps?

            if hasattr(self._executor, "after_action_selection"):
                if hasattr(self._executor, "_counts"):
                    loop_type = "evaluator" if self._executor._evaluator else "executor"
                    total_steps_before_current_episode = self._executor._counts[
                        f"{loop_type}_steps"
                    ].numpy()
                else:
                    total_steps_before_current_episode = self._counter.get_counts().get(
                        "executor_steps", 0
                    )
                current_step_t = total_steps_before_current_episode + episode_steps
                self._executor.after_action_selection(current_step_t)

            self._compute_step_statistics(rewards)

            for agent, reward in rewards.items():
                episode_returns[agent] = episode_returns[agent] + reward

        self._compute_episode_statistics(
            episode_returns,
            episode_steps,
            start_time,
        )

        if self._get_running_stats():
            return self._get_running_stats()
        else:
            counts = self.record_counts(episode_steps)

            # Collect the results and combine with counts.
            steps_per_second = episode_steps / (time.time() - start_time)
            result = {
                "episode_length": episode_steps,
                "mean_episode_return": np.mean(list(episode_returns.values())),
                "steps_per_second": steps_per_second,
            }
            result.update(counts)
            return result

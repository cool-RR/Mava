from mava.components.jax import Component
from chex import dataclass
from mava.wrappers.offline_environment_logger import MAOfflineEnvironmentSequenceLogger
from mava.core_jax import SystemExecutor


@dataclass
class EvaluatorOfflineLoggingConfig:
    offline_sequence_length: int = 1000
    offline_sequence_period: int = 100
    offline_logdir: str = "~./offline_env_logs"
    offline_label: str = "offline_logger"
    offline_min_sequences_per_file: int = 1000


class EvaluatorOfflineLogging(Component):
    def __init__(
        self,
        config: EvaluatorOfflineLoggingConfig = EvaluatorOfflineLoggingConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_execution_init_end(self, executor: SystemExecutor):
        print("in exec init")
        if executor.store.is_evaluator:
            print(executor.store)
            executor.store.environment_loop._environment = (
                MAOfflineEnvironmentSequenceLogger(
                    executor.store.environment_loop._environment,
                    self.config.offline_sequence_length,
                    self.config.offline_sequence_period,
                    self.config.offline_logdir,
                    self.config.offline_label,
                    self.config.offline_min_sequences_per_file,
                )
            )

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "evaluator_offline_logging"  # for creating system lowercase underscore

    @staticmethod
    def config_class():
        return EvaluatorOfflineLoggingConfig

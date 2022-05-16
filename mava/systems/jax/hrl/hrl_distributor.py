from mava.components.jax.building.distributor import Distributor
from mava.core_jax import SystemBuilder
from mava.systems.jax import Launcher
from mava.systems.jax.launcher import NodeType


class HrlDistributor(Distributor):
    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.store.program = Launcher(
            multi_process=self.config.multi_process,
            nodes_on_gpu=self.config.nodes_on_gpu,
            name=self.config.distributor_name,
        )

        # reverb
        hl_data_server = builder.store.program.add(
            builder.data_server,
            node_type=NodeType.reverb,
            name="data_server",
        )

        ll_data_server = builder.store.program.add(
            builder.data_server,
            node_type=NodeType.reverb,
            name="data_server",
        )
        data_server = (hl_data_server, ll_data_server)

        # variable server node
        parameter_server = builder.store.program.add(
            builder.parameter_server,
            node_type=NodeType.corrier,
            name="parameter_server",
        )

        # executor nodes
        for executor_id in range(self.config.num_executors):
            builder.store.program.add(
                builder.executor,
                [f"executor_{executor_id}", data_server, parameter_server],
                node_type=NodeType.corrier,
                name="executor",
            )

        if self.config.run_evaluator:
            # evaluator node
            builder.store.program.add(
                builder.executor,
                ["evaluator", data_server, parameter_server],
                node_type=NodeType.corrier,
                name="evaluator",
            )

        # trainer nodes
        for trainer_id in builder.store.trainer_networks.keys():
            builder.store.program.add(
                builder.trainer,
                [trainer_id, data_server, parameter_server],
                node_type=NodeType.corrier,
                name="trainer",
            )

        if not self.config.multi_process:
            builder.store.system_build = builder.store.program.get_nodes()

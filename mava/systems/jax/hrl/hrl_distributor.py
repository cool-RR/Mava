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
        print("TYPE BUILDER!!!!", type(builder))

        builder.store.program = Launcher(
            multi_process=self.config.multi_process,
            nodes_on_gpu=self.config.nodes_on_gpu,
            name=self.config.distributor_name,
        )

        # experience nodes
        builder.store.spec_key = "hl"
        hl_data_server = builder.store.program.add(
            builder.data_server,
            node_type=NodeType.reverb,
            name="hl_data_server",
        )

        builder.store.spec_key = "ll"
        ll_data_server = builder.store.program.add(
            builder.data_server,
            node_type=NodeType.reverb,
            name="ll_data_server",
        )
        data_servers = (hl_data_server, ll_data_server)

        # variable server nodes
        builder.store.net_level_key = "hl"
        hl_parameter_server = builder.store.program.add(
            builder.parameter_server,
            node_type=NodeType.corrier,
            name="hl_parameter_server",
        )

        builder.store.net_level_key = "ll"
        ll_parameter_server = builder.store.program.add(
            builder.parameter_server,
            node_type=NodeType.corrier,
            name="ll_parameter_server",
        )

        parameter_servers = (hl_parameter_server, ll_parameter_server)
        # executor nodes
        for executor_id in range(self.config.num_executors):
            builder.store.program.add(
                builder.executor,
                [f"executor_{executor_id}", data_servers, parameter_servers],
                node_type=NodeType.corrier,
                name="executor",
            )

        if self.config.run_evaluator:
            # evaluator node
            builder.store.program.add(
                builder.executor,
                ["evaluator", data_servers, parameter_servers],
                node_type=NodeType.corrier,
                name="evaluator",
            )

        # trainer nodes
        for trainer_id in builder.store.trainer_networks.keys():
            builder.store.program.add(
                builder.trainer,
                [trainer_id, hl_data_server, hl_parameter_server],
                node_type=NodeType.corrier,
                name="hl_trainer",
            )

            builder.store.program.add(
                builder.trainer,
                [trainer_id, ll_data_server, ll_parameter_server],
                node_type=NodeType.corrier,
                name="ll_trainer",
            )

        if not self.config.multi_process:
            builder.store.system_build = builder.store.program.get_nodes()

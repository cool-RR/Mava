from mava.components.jax.building import ParallelSequenceAdder
from mava.core_jax import SystemBuilder
from mava.adders import reverb as reverb_adders


class HrlParallelSequenceAdder(ParallelSequenceAdder):
    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        assert not hasattr(builder.store, "adder_priority_fn")
        assert isinstance(builder.store.data_server_client, tuple)

        # Create custom priority functons for the adder
        priority_fns = {
            table_key: lambda x: 1.0
            for table_key in builder.store.table_network_config.keys()
        }

        adders = []
        for data_server_client in builder.store.data_server_client:
            adders.append(reverb_adders.ParallelSequenceAdder(
                priority_fns=priority_fns,
                client=data_server_client,
                net_ids_to_keys=builder.store.unique_net_keys,
                sequence_length=self.config.sequence_length,
                table_network_config=builder.store.table_network_config,
                period=self.config.period,
                use_next_extras=self.config.use_next_extras,
            ))

        builder.store.adder = adders

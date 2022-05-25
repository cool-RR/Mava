from mava.components.jax.building import ExecutorParameterClient, TrainerParameterClient
from mava.core_jax import SystemBuilder
from mava.systems.jax import ParameterClient


class HrlExecutorParameterClient(ExecutorParameterClient):
    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        parameter_clients = []
        for net_level_key in ["hl", "ll"]:
            # Create policy parameters
            params = {}
            get_keys = []
            net_type_key = "networks"
            for agent_net_key in builder.store.networks[net_type_key].keys():
                # Executor gets both high level and low level params
                param_key = f"{net_type_key}-{agent_net_key}-{net_level_key}"
                params[param_key] = builder.store.networks[net_type_key][agent_net_key][
                    net_level_key
                ].params
                get_keys.append(param_key)

            count_names, params = self._set_up_count_parameters(params=params)
            get_keys.extend(count_names)

            builder.store.executor_counts = {name: params[name] for name in count_names}

            set_keys = get_keys.copy()
            # Executors should only be able to update relevant params.
            if builder.store.is_evaluator is True:
                set_keys = [x for x in set_keys if x.startswith("evaluator")]
            else:
                set_keys = [x for x in set_keys if x.startswith("executor")]

            parameter_client = None
            if builder.store.parameter_server_client:
                # Create parameter client
                parameter_client = ParameterClient(
                    client=builder.store.parameter_server_client[net_level_key],
                    parameters=params,
                    get_keys=get_keys,
                    set_keys=set_keys,  # why do executors need to set any params?
                    update_period=self.config.executor_parameter_update_period,
                )

                # Make sure not to use a random policy after checkpoint restoration by
                # assigning parameters before running the environment loop.
                parameter_client.get_and_wait()

            parameter_clients.append(parameter_client)

        builder.store.executor_parameter_client = parameter_clients


class HrlTrainerParameterClient(TrainerParameterClient):
    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        # Create parameter client
        params = {}
        set_keys = []
        get_keys = []
        # TODO (dries): Only add the networks this trainer is working with.
        # Not all of them.
        trainer_networks = builder.store.trainer_networks[builder.store.trainer_id]
        for net_type_key in builder.store.networks.keys():
            for net_key in builder.store.networks[net_type_key].keys():
                param_key = f"{net_type_key}-{net_key}-{builder.store.net_level_key}"
                params[param_key] = builder.store.networks[net_type_key][net_key][
                    builder.store.net_level_key
                ].params
                if net_key in set(trainer_networks):
                    set_keys.append(param_key)
                else:
                    get_keys.append(param_key)

        # Add the optimizers to the variable server.
        # TODO (dries): Adjust this if using policy and critic optimizers.
        # TODO (dries): Add this back if we want the optimizer_state to
        # be store in the variable source. However some code might
        # need to be moved around as the builder currently does not
        # have access to the opt_states yet.
        # params["optimizer_state"] = trainer.store.opt_states

        count_names, params = self._set_up_count_parameters(params=params)

        get_keys.extend(count_names)
        builder.store.trainer_counts = {name: params[name] for name in count_names}

        # Create parameter client
        parameter_client = None
        if builder.store.parameter_server_client:
            parameter_client = ParameterClient(
                client=builder.store.parameter_server_client,
                parameters=params,
                get_keys=get_keys,
                set_keys=set_keys,
            )

            # Get all the initial parameters
            parameter_client.get_all_and_wait()

        builder.store.trainer_parameter_client = parameter_client

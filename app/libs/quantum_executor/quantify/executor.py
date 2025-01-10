# This code is part of Tergite
#
# (C) Axel Andersson (2022)
# (C) Martin Ahindura (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Refactored by Martin Ahindura (2024)
# Refactored by Stefan Hill (2024)


import copy
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Union

import qblox_instruments
import rich
from qcodes import Instrument, find_or_create_instrument
from quantify_scheduler.backends.qblox.helpers import generate_port_clock_to_device_map
from quantify_scheduler.backends.qblox_backend import hardware_compile
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components import (
    InstrumentCoordinatorComponentBase,
    generic,
)
from quantify_scheduler.instrument_coordinator.components.generic import (
    GenericInstrumentCoordinatorComponent,
)
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent

from app.libs.quantum_executor.base.executor import QuantumExecutor
from app.libs.quantum_executor.quantify.experiment import QuantifyExperiment
from app.libs.quantum_executor.utils.config import (
    ClusterModuleType,
    QuantifyExecutorConfig,
)
from app.libs.quantum_executor.base.experiment import NativeQobjConfig
from app.libs.quantum_executor.utils.logger import ExperimentLogger

_QBLOX_CLUSTER_TYPE_MAP: Dict[ClusterModuleType, qblox_instruments.ClusterType] = {
    ClusterModuleType.QCM: qblox_instruments.ClusterType.CLUSTER_QCM,
    ClusterModuleType.QRM: qblox_instruments.ClusterType.CLUSTER_QRM,
    ClusterModuleType.QCM_RF: qblox_instruments.ClusterType.CLUSTER_QCM_RF,
    ClusterModuleType.QRM_RF: qblox_instruments.ClusterType.CLUSTER_QRM_RF,
}

_MODULE_NAME_REGEX = re.compile(r".*_module(\d+)$")


class QuantifyExecutor(QuantumExecutor):
    """The controller of the hardware that executes the quantum jobs"""

    _coordinator = find_or_create_instrument(
        InstrumentCoordinator,
        "tergite_quantum_executor",
        # the default generic icc is important for QCoDeS commands that are run generically
        # when creating a generic QCoDeS instrument
        add_default_generic_icc=True,
    )

    # heap memory, so that instrument drivers do not get garbage collected
    shared_mem = dict()

    def __init__(self, config_file: Union[str, bytes, os.PathLike]):
        conf = QuantifyExecutorConfig.from_yaml(config_file)
        self.quantify_config = conf.to_quantify()
        self.hardware_map = {
            clock: port
            for (port, clock), instrument in generate_port_clock_to_device_map(
                self.quantify_config
            ).items()
        }
        super().__init__(
            experiment_cls=QuantifyExperiment, hardware_map=self.hardware_map
        )

        # load clusters
        for cluster in conf.clusters:
            dummy_cfg: Optional[Dict[int, qblox_instruments.ClusterType]] = None
            if cluster.is_dummy:
                dummy_cfg = {
                    # No checks or try catches because the config is expected to be in the right format
                    int(
                        _MODULE_NAME_REGEX.match(module.name).group(1)
                    ): _QBLOX_CLUSTER_TYPE_MAP[module.instrument_type]
                    for module in cluster.modules
                }
            # We only support qblox_instruments.Cluster for now. Pulsar and any other native interfaces were dropped
            # because they cause a chaotic configuration.
            # The Cluster was also the only one documented on quantify-scheduler docs at the time of the refactor
            # https://quantify-os.org/docs/quantify-scheduler/dev/reference/qblox/Cluster.html
            device = find_or_create_instrument(
                qblox_instruments.Cluster,
                name=cluster.name,
                identifier=cluster.instrument_address,
                dummy_cfg=dummy_cfg,
            )
            QuantifyExecutor.shared_mem[device.name] = device
            rich.print(f"Instantiated Cluster driver for '{cluster.name}'")
            _add_component_if_not_exists(
                coordinator=QuantifyExecutor._coordinator,
                component_type=ClusterComponent,
                device=device,
            )

        # load generic QCoDes instruments
        for instrument in conf.generic_qcodes_instruments:
            # instantiate the device
            driver = import_python_object_from_string(
                instrument.instrument_driver.import_path
            )
            device_name = instrument.instrument_driver.kwargs.pop(
                "name", generic.DEFAULT_NAME
            )
            device = find_or_create_instrument(
                driver, device_name, **instrument.instrument_driver.kwargs
            )
            _set_parameters(device, instrument.parameters)
            QuantifyExecutor.shared_mem[device.name] = device
            rich.print(
                f"Instantiated {instrument.instrument_driver.import_path.split('.')[-1]} driver for '{instrument.name}'"
            )

            _add_component_if_not_exists(
                coordinator=QuantifyExecutor._coordinator,
                component_type=GenericInstrumentCoordinatorComponent,
                device=device,
            )

    def _run_native(
        self,
        experiment: QuantifyExperiment,
        /,
        *,
        native_config: NativeQobjConfig,
        logger: ExperimentLogger,
    ):
        QuantifyExecutor._coordinator.stop()

        # compile to hardware
        # TODO: Here, we can use the new @timer decorator in the benchmarking package
        t1 = datetime.now()
        absolute_timed_schedule = determine_absolute_timing(
            copy.deepcopy(experiment.schedule)
        )
        compiled_schedule = hardware_compile(
            schedule=absolute_timed_schedule, hardware_cfg=self.quantify_config
        )

        t2 = datetime.now()
        print(t2 - t1, "DURATION OF COMPILING")

        # log the sequencer assembler programs and the schedule timing table
        logger.log_Q1ASM_programs(compiled_schedule)
        logger.log_schedule(compiled_schedule)

        # upload schedule to instruments & arm sequencers
        self._coordinator.prepare(compiled_schedule)

        # start experiment
        # TODO: Here, we can use the new @timer decorator from the benchmarking package
        t3 = datetime.now()
        QuantifyExecutor._coordinator.start()

        # wait for program to finish and return acquisition
        # TODO: What is the return type of retrieve_acquisition()?
        results = self._coordinator.retrieve_acquisition()
        print(f"{results=}")
        t4 = datetime.now()
        print(t4 - t3, "DURATION OF MEASURING")
        return results

    @classmethod
    def close(cls):
        """Closes the QuantumExecutor associated with this name"""
        cls._coordinator.close_all()


def _add_component_if_not_exists(
    coordinator: InstrumentCoordinator,
    component_type: type[InstrumentCoordinatorComponentBase],
    device: Instrument,
):
    """Adds a component for the given device to the coordinator

    Args:
        coordinator: the instrument coordinator to add component to
        component_type: the type of component
        device: the instrument for the given component
    """
    try:
        component = coordinator.get_component(f"ic_{device.name}")
    except KeyError:
        component = component_type(device)

    try:
        coordinator.add_component(component)
        rich.print(
            f"Added '{component.name}' to instrument coordinator '{coordinator.name}'"
        )
    except ValueError:
        # ignore if component is already added
        pass


def _set_parameters(device: Instrument, parameters: Dict[str, Any]):
    """Set the parameters of a QCoDeS device

    Args:
        device: the QCoDeS device whose parameters are to be set
        parameters: the dictionary of parameter names and values
    """

    for command, value in parameters.items():
        try:
            # Setting parameters is done by calling them as commands
            # https://microsoft.github.io/Qcodes/examples/15_minutes_to_QCoDeS.html#Example-of-setting-and-getting-parameters
            qcodes_command = getattr(device, command)
            qcodes_command(value)
            rich.print(f"Set '{command}' to {value}")
        except (AttributeError, TypeError):
            # ignore invalid parameters
            pass

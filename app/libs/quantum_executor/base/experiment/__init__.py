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

import abc
import copy
from dataclasses import dataclass
from functools import cached_property
from typing import FrozenSet, List, Optional, Dict, TYPE_CHECKING

import retworkx as rx
from pandas import DataFrame
from qiskit.qobj import (
    PulseQobjConfig,
    QobjExperimentHeader,
    PulseQobjExperiment,
    PulseQobjInstruction,
)

from app.libs.quantum_executor.base.utils import NativeQobjConfig
from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.base.instruction import (
    Instruction,
    DelayInstruction,
    ParamPulseInstruction,
    PhaseInstruction,
    FreqInstruction,
    PulseLibInstruction,
    AcquireInstruction,
)
from app.libs.quantum_executor.utils.general import flatten_list, ceil4

if TYPE_CHECKING:
    from app.libs.quantum_executor.base.utils import NativeQobjConfig


@dataclass(frozen=True)
class NativeExperiment(abc.ABC):
    header: QobjExperimentHeader
    instructions: List[Instruction]
    config: PulseQobjConfig
    channels: FrozenSet[Channel]
    buffer_time: float = 0.0

    @cached_property
    def dag(self: "NativeExperiment"):
        dag = rx.PyDiGraph(check_cycle=True, multigraph=False)

        prev_index = dict()
        for j in sorted(self.instructions, key=lambda j: j.t0):
            if j.channel not in prev_index.keys():
                # add the first non-trivial instruction on the channel
                prev_index[j.channel] = dag.add_node(j)
            else:
                # get node index of previous instruction
                i = dag[prev_index[j.channel]]

                # add the next instruction
                prev_index[j.channel] = dag.add_child(
                    parent=prev_index[j.channel], obj=j, edge=j.t0 - (i.t0 + i.duration)
                )

        return dag

    @property
    @abc.abstractmethod
    def schedule(self):
        pass

    @property
    def timing_table(self: "NativeExperiment") -> DataFrame:
        df = self.schedule.timing_table.data
        df.sort_values("abs_time", inplace=True)
        return df

    @classmethod
    def from_qobj_expt(
        cls,
        expt: PulseQobjExperiment,
        name: str,
        qobj_config: PulseQobjConfig,
        native_config: NativeQobjConfig,
        hardware_map: Optional[Dict[str, str]],
    ) -> "NativeExperiment":
        """Converts PulseQobjExperiment to native experiment

        Args:
            expt: the pulse qobject experiment to translate
            name: the name of the experiment
            qobj_config: the pulse qobject config
            native_config: the native config for the qobj
            hardware_map: the map of the real/simulated device to the logical definitions

        Returns:
            the QiskitDynamicsExperiment corresponding to the PulseQobj
        """
        header = copy.copy(expt.header)
        header.name = name
        inst_nested_list = (
            _extract_instructions(
                qobj_inst=inst,
                config=qobj_config,
                native_config=native_config,
                hardware_map=hardware_map,
            )
            for inst in expt.instructions
        )
        native_instructions = flatten_list(inst_nested_list)

        return cls(
            header=header,
            instructions=native_instructions,
            config=qobj_config,
            channels=frozenset(
                Channel(
                    clock=i.channel,
                    frequency=0.0,
                )
                for i in native_instructions
            ),
        )


def _extract_instructions(
    qobj_inst: PulseQobjInstruction,
    config: PulseQobjConfig,
    native_config: NativeQobjConfig,
    hardware_map: Dict[str, str] = None,
) -> List[Instruction]:
    """Extracts tergite-specific instructions from the PulseQobjInstruction

    Args:
        qobj_inst: the PulseQobjInstruction from which instructions are to be extracted
        config: config of the pulse qobject
        native_config: the native config for the qobj
        hardware_map: the map describing the layout of the quantum device

    Returns:
        list of tergite-specific instructions
    """
    if hardware_map is None:
        hardware_map = {}

    name = qobj_inst.name
    t0 = ceil4(qobj_inst.t0) * 1e-9
    channel = qobj_inst.ch

    if name == "delay":
        return [
            DelayInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=ceil4(qobj_inst.duration) * 1e-9,
            )
        ]

    if name == "parametric_pulse":
        return [
            ParamPulseInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=ceil4(qobj_inst.parameters["duration"]) * 1e-9,
                pulse_shape=qobj_inst.pulse_shape,
                parameters=qobj_inst.parameters,
            )
        ]

    if name in ("setp", "fc"):  # "shiftf" is not working
        return [
            PhaseInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=0.0,
                phase=qobj_inst.phase,
            )
        ]

    if qobj_inst.name in ("setf",):  # "shiftf" is not working
        return [
            FreqInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=0.0,
                frequency=qobj_inst.frequency * 1e9,
            )
        ]

    if qobj_inst.name in config.pulse_library:
        return [
            PulseLibInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                # FIXME: pulse_library seems to be a list but is accessed here as a dict
                duration=ceil4(config.pulse_library[name].shape[0]) * 1e-9,
            )
        ]

    if name == "acquire":
        return [
            AcquireInstruction(
                name=name,
                t0=t0,
                channel=f"m{qubit_idx}",
                port=hardware_map.get(f"m{qubit_idx}", name),
                duration=ceil4(qobj_inst.duration) * 1e-9,
                memory_slot=qobj_inst.memory_slot[n],
                protocol=native_config.protocol.value,
                acq_return_type=native_config.acq_return_type,
                bin_mode=native_config.bin_mode,
            )
            for n, qubit_idx in enumerate(qobj_inst.qubits)
        ]

    raise RuntimeError(f"No mapping for PulseQobjInstruction {qobj_inst}")

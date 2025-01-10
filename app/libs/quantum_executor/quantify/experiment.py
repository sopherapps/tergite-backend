# This code is part of Tergite
#
# (C) Axel Andersson (2022)
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
from dataclasses import dataclass
from typing import Optional, Dict, List, Type

import retworkx as rx
from qiskit.qobj import PulseQobjExperiment, PulseQobjConfig, PulseQobjInstruction
from quantify_scheduler import Schedule

from app.libs.quantum_executor.base.experiment import NativeExperiment
from app.libs.quantum_executor.base.utils import NativeQobjConfig
from .program import QuantifyProgram
from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.utils.general import rot_left, flatten_list, ceil4
from app.libs.quantum_executor.base.instruction import Instruction
from .instruction import (
    InitialObjectInstruction,
    AcquireInstruction,
    DelayInstruction,
    FreqInstruction,
    PhaseInstruction,
    ParamPulseInstruction,
    PulseLibInstruction,
)
from ..base.experiment.utils import copy_header_with

# FIXME: Why is this initial object hard coded here?
initial_object = InitialObjectInstruction()

# Map name => Instruction
_INSTRUCTION_MAP: Dict[str, Type[Instruction]] = {
    "setf": FreqInstruction,
    "setp": PhaseInstruction,
    "fc": PhaseInstruction,
    "delay": DelayInstruction,
    "parametric_pulse": AcquireInstruction,
}


@dataclass(frozen=True)
class QuantifyExperiment(NativeExperiment):
    @property
    def schedule(self: "QuantifyExperiment") -> Schedule:
        prog = QuantifyProgram(
            name=self.header.name,
            channels=self.channels,
            config=self.config,
        )
        prog.schedule_operation(initial_object, ref_op=None, rel_time=0.0)

        wccs = rx.weakly_connected_components(self.dag)

        for wcc in wccs:
            wcc_nodes = list(sorted(list(wcc)))

            # if the channel contains a single instruction and that instruction is a delay,
            # then do not schedule any operations on that channel
            if len(wcc_nodes) == 1:
                if self.dag[wcc_nodes[0]].name == "delay":
                    print()
                    print("NO DELAY")
                    print()
                    continue

            # else, schedudle the instructions on the channels
            for n, idx in enumerate(wcc_nodes):
                ref_idx = next(iter(rot_left(reversed(wcc_nodes[: n + 1]), 1)))
                if ref_idx == idx:
                    ref_op = initial_object.label
                    rel_time = self.buffer_time
                else:
                    ref_op = self.dag[ref_idx].label
                    rel_time = self.dag.get_edge_data(ref_idx, idx) + 4e-9

                prog.schedule_operation(
                    self.dag[idx],
                    rel_time=rel_time,
                    ref_op=ref_op,
                )

        return prog.compiled_schedule

    @classmethod
    def from_qobj_expt(
        cls,
        expt: PulseQobjExperiment,
        name: str,
        qobj_config: PulseQobjConfig,
        native_config: NativeQobjConfig,
        hardware_map: Optional[Dict[str, str]],
    ) -> "QuantifyExperiment":
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
        header = copy_header_with(expt.header, name=name)
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

    try:
        cls = _INSTRUCTION_MAP[qobj_inst.name]
    except KeyError as exp:
        if qobj_inst.name in config.pulse_library:
            cls = PulseLibInstruction
        else:
            raise RuntimeError(
                f"No mapping for PulseQobjInstruction {qobj_inst}.\n{exp}"
            )

    return cls.list_from_qobj_inst(
        qobj_inst, config=config, native_config=native_config, hardware_map=hardware_map
    )

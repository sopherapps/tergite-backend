# This code is part of Tergite
#
# (C) Stefan Hill (2024)
# (C) Pontus Vikstål (2024)
# (C) Chalmers Next Labs (2024)
# (C) Martin Ahindura (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import List, Dict, Type, Tuple, Optional

from qiskit.qobj import PulseQobjExperiment, PulseQobjConfig, PulseQobjInstruction

from app.libs.quantum_executor.base.experiment import NativeExperiment

from qiskit.pulse.schedule import Schedule

from app.libs.quantum_executor.base.experiment.utils import copy_header_with
from .instruction import (
    GaussianPlay,
    WacqtCZPlay,
    QiskitDynamicsInstruction,
    SetFrequency,
    ShiftFrequency,
    SetPhase,
    ShiftPhase,
    Delay,
    Acquire,
)

# Map (name, pulse_shape) => Instruction
_INSTRUCTION_PULSE_MAP: Dict[
    Tuple[str, Optional[str]], Type[QiskitDynamicsInstruction]
] = {
    ("setf", None): SetFrequency,
    ("shiftf", None): ShiftFrequency,
    ("setp", None): SetPhase,
    ("fc", None): ShiftPhase,
    ("delay", None): Delay,
    ("parametric_pulse", "constant"): Acquire,
    ("parametric_pulse", "gaussian"): GaussianPlay,
    ("parametric_pulse", "wacqt_cz_gate_pulse"): WacqtCZPlay,
}


@dataclass(frozen=True)
class QiskitDynamicsExperiment(NativeExperiment):
    raw_schedule: Schedule = None
    instructions: List[QiskitDynamicsInstruction]

    @cached_property
    def schedule(self) -> "Schedule":
        return self.raw_schedule

    @classmethod
    def from_qobj_expt(
        cls,
        expt: PulseQobjExperiment,
        name: str,
        qobj_config: PulseQobjConfig,
    ) -> "QiskitDynamicsExperiment":
        """Converts PulseQobjExperiment to qiskit dynamics experiment

        Args:
            expt: the pulse qobject experiment to translate
            name: the name of the experiment
            qobj_config: the pulse qobject config

        Returns:
            the QiskitDynamicsExperiment corresponding to the PulseQobj
        """
        header = copy_header_with(expt.header, name=name)
        timestamp: str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        raw_schedule = Schedule(name=f"open-pulse-generated-{timestamp}")

        native_instructions: List[QiskitDynamicsInstruction] = []
        qobj_instructions: List[PulseQobjInstruction] = expt.instructions

        for inst in qobj_instructions:
            native_inst = _to_native_instruction(inst)
            native_instructions.append(native_inst)
            raw_schedule.insert(inst.t0, native_inst)

        return cls(
            header=header,
            instructions=native_instructions,
            config=qobj_config,
            channels=frozenset(),
            raw_schedule=raw_schedule,
        )


def _to_native_instruction(
    qobj_inst: PulseQobjInstruction,
) -> QiskitDynamicsInstruction:
    """Extracts qiskit pulse instruction from the PulseQobjInstruction

    Args:
        qobj_inst: the PulseQobjInstruction from which instructions are to be extracted

    Returns:
        the native qiskit instruction
    """
    name = qobj_inst.name
    pulse_shape = qobj_inst.pulse_shape

    try:
        native_inst_cls = _INSTRUCTION_PULSE_MAP[(name, pulse_shape)]
        return native_inst_cls.from_qobj(qobj_inst)
    except IndexError as exp:
        raise RuntimeError(f"No mapping for PulseQobjInstruction {qobj_inst}.\n {exp}")

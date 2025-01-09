# This code is part of Tergite
#
# (C) Stefan Hill (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import FrozenSet

from qiskit.pulse import (
    SetPhase,
    SetFrequency,
    ShiftPhase,
    ShiftFrequency,
    Play,
    Delay,
    Acquire,
    MemorySlot,
    DriveChannel,
    AcquireChannel,
    Instruction,
)
from qiskit.pulse.library import Gaussian
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import PulseQobjConfig

from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.base.instruction import (
    Instruction as OpenPulseInstruction,
)
from app.libs.quantum_executor.utils.logger import ExperimentLogger

# TODO SIM: If we use the transpile function, this whole Experiment object might become redundant


# @dataclass(frozen=True)
class QiskitDynamicsProgram:
    def __init__(
        self,
        name: str = "",
        channels: FrozenSet[Channel] = None,
        config: PulseQobjConfig = None,
        logger: ExperimentLogger = None,
    ):
        self.name = name
        self.channels = channels
        self.config = config
        self.logger = logger
        self.schedule = Schedule(name=self.name)

    def get_channel(self, instruction: OpenPulseInstruction, /) -> "Channel":
        return next(filter(lambda ch: instruction.channel == ch.clock, self.channels))

    def update_frame(self, instruction: OpenPulseInstruction, /):
        # relative phase change
        if instruction.name == "fc":
            self.get_channel(instruction).phase += instruction.phase

        # absolute phase change
        elif instruction.name == "setp":
            self.get_channel(instruction).phase = instruction.phase

        # relative frequency change
        elif instruction.name == "shiftf":
            self.get_channel(instruction).frequency += instruction.frequency

        # absolute frequency change
        elif instruction.name == "setf":
            self.get_channel(instruction).frequency = instruction.frequency

        else:
            raise RuntimeError(f"Unable to execute command {instruction}.")

    def schedule_operation(
        self,
        instruction: "OpenPulseInstruction",
        /,
        is_measurement: bool = False,
    ):
        operation: "Instruction"

        # TODO SIM: This is hardcoded right now, we have to pass it manually to some extend
        dt = 1e-9

        channel = int(instruction.channel.strip("d").strip("readout").strip("m"))

        frequency_operation_map_ = {"setf": SetFrequency, "shiftf": ShiftFrequency}
        phase_operation_map_ = {"setp": SetPhase, "fc": ShiftPhase}

        if instruction.name in frequency_operation_map_.keys() and not is_measurement:
            operation = frequency_operation_map_[instruction.name](
                instruction.frequency, DriveChannel(channel)
            )
            self.update_frame(instruction)

        elif instruction.name in phase_operation_map_.keys() and not is_measurement:
            operation = phase_operation_map_[instruction.name](
                instruction.phase, DriveChannel(channel)
            )
            self.update_frame(instruction)

        elif instruction.name == "delay":
            operation = Delay(instruction.parameters["duration"], DriveChannel(channel))

        elif (
            instruction.name == "parametric_pulse"
            and instruction.pulse_shape == "constant"
        ):
            operation = Acquire(
                1,  # set duration to 0, because it does not matter
                AcquireChannel(channel),
                MemorySlot(channel),
            )
            self.get_channel(
                instruction
            ).acquisitions += 1  # increment no. of acquisitions

        elif (
            instruction.name == "parametric_pulse"
            and instruction.pulse_shape == "gaussian"
        ):
            gauss = Gaussian(
                instruction.parameters["duration"],
                instruction.parameters["amp"],
                instruction.parameters["sigma"],
            )
            operation = Play(gauss, DriveChannel(channel))

        elif is_measurement:
            return

        try:
            self.schedule = self.schedule.insert(int(instruction.t0 * 10e9), operation)
            return

        except UnboundLocalError:
            raise RuntimeError(f"Unable to schedule operation {instruction}.")

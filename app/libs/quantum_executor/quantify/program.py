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

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import qiskit.pulse
from quantify_scheduler import Operation, Schedule
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.resources import ClockResource

from app.libs.quantum_executor.base.program import BaseProgram
from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.base.instruction import Instruction


@dataclass(frozen=True)
class QuantifyProgram(BaseProgram):
    @cached_property
    def schedule(self: "QuantifyProgram") -> "Schedule":
        return Schedule(name=self.name, repetitions=self.config.shots)

    @cached_property
    def compiled_schedule(self: "QuantifyProgram") -> "Schedule":
        for channel in self.channels:
            clock = ClockResource(name=channel.clock, freq=channel.frequency)
            self.schedule.add_resource(clock)

        return determine_absolute_timing(self.schedule)

    def get_channel(self: "QuantifyProgram", instruction: Instruction, /) -> "Channel":
        return next(filter(lambda ch: instruction.channel == ch.clock, self.channels))

    def numerical_pulse(
        self: "QuantifyProgram", instruction: Instruction, /, *, waveform: np.ndarray
    ) -> "Operation":
        waveform *= np.exp(1.0j * self.get_channel(instruction).phase)
        operation = Operation(name=instruction.unique_name)
        operation.data.update(
            {
                "pulse_info": [
                    {
                        "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
                        "samples": waveform.tolist(),
                        "t_samples": np.linspace(
                            0, instruction.duration, len(waveform)
                        ).tolist(),
                        "duration": instruction.duration,
                        "interpolation": "linear",
                        "clock": instruction.channel,
                        "port": instruction.port,
                        "t0": 0.0,
                    }
                ],
            }
        )
        operation._update()
        return operation

    def update_frame(self: "QuantifyProgram", instruction: Instruction, /):
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
        self: "QuantifyProgram",
        instruction: Instruction,
        /,
        *,
        rel_time: float,
        ref_op: str,
    ):
        # -----------------------------------------------------------------
        if instruction.name in {"setp", "setf", "fc", "shiftf"}:
            self.update_frame(instruction)
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        if instruction.name in {
            "setp",
            "setf",
            "fc",
            "shiftf",
            "initial_object",
            "delay",
        }:
            operation = Operation(name=instruction.unique_name)
            operation.data.update(
                {
                    "pulse_info": [
                        {
                            "wf_func": None,
                            "t0": 0.0,
                            "duration": instruction.duration,
                            "clock": instruction.channel,
                            "port": None,
                        }
                    ]
                }
            )
            operation._update()
        # -----------------------------------------------------------------
        elif instruction.name == "acquire":
            if instruction.protocol == "SSBIntegrationComplex":
                waveform_i = {
                    "port": instruction.port,
                    "clock": instruction.channel,
                    "t0": 0.0,
                    "duration": instruction.duration,
                    "wf_func": "quantify_scheduler.waveforms.square",
                    "amp": 1,
                }
                waveform_q = {
                    "port": instruction.port,
                    "clock": instruction.channel,
                    "t0": 0.0,
                    "duration": instruction.duration,
                    "wf_func": "quantify_scheduler.waveforms.square",
                    "amp": 1j,
                }
                weights = [waveform_i, waveform_q]
            elif instruction.protocol == "trace":
                weights = []

            else:
                raise RuntimeError(
                    f"Cannot schedule acquisition with unknown protocol {instruction.protocol}."
                )

            operation = Operation(name=instruction.unique_name)
            operation.data.update(
                {
                    "acquisition_info": [
                        {
                            "waveforms": weights,
                            "t0": 0.0,
                            "clock": instruction.channel,
                            "port": instruction.port,
                            "duration": instruction.duration,
                            "phase": 0.0,
                            # "acq_channel": instruction.memory_slot, # TODO: Fix deranged memory slot readout
                            "acq_channel": int(
                                instruction.channel[1:]
                            ),  # FIXME, hardcoded single character parsing
                            "acq_index": self.get_channel(instruction).acquisitions,
                            "bin_mode": instruction.bin_mode,
                            "acq_return_type": instruction.acq_return_type,
                            "protocol": instruction.protocol,
                        }
                    ]
                }
            )

            operation._update()
            self.get_channel(
                instruction
            ).acquisitions += 1  # increment no. of acquisitions
        # -----------------------------------------------------------------
        elif instruction.name == "parametric_pulse":
            wf_fn = getattr(
                qiskit.pulse.library.discrete, str.lower(instruction.pulse_shape)
            )
            waveform = wf_fn(**instruction.parameters).samples
            operation = self.numerical_pulse(instruction, waveform=waveform)
        # -----------------------------------------------------------------
        elif instruction.name in self.config.pulse_library:
            waveform = self.config.pulse_library[instruction.name]
            operation = self.numerical_pulse(instruction, waveform=waveform)
        # -----------------------------------------------------------------
        else:
            raise RuntimeError(f"Unable to schedule operation {instruction}.")
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        self.schedule.add(
            ref_op=ref_op,
            ref_pt="end",
            ref_pt_new="start",
            rel_time=rel_time,
            label=instruction.label,
            operation=operation,
        )

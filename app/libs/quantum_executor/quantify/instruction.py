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
from typing import List, Optional, Dict, Any

from qiskit.qobj import PulseQobjInstruction, PulseQobjConfig

from app.libs.quantum_executor.base.instruction import Instruction
from app.libs.quantum_executor.base.utils import NativeQobjConfig
from app.libs.quantum_executor.utils.general import ceil4


class InitialObjectInstruction(Instruction):
    __slots__ = ()

    def __init__(self, t0=0.0, channel="cl0.baseband", duration=0.0, **kwargs):
        kwargs["name"] = "initial_object"
        super().__init__(t0=t0, channel=channel, duration=duration, **kwargs)

    @classmethod
    def list_from_qobj_inst(
        cls, qobj_inst: PulseQobjInstruction, **kwargs
    ) -> List["InitialObjectInstruction"]:
        t0 = ceil4(qobj_inst.t0) * 1e-9

        return [
            cls(
                t0=t0,
                channel=qobj_inst.ch,
                duration=qobj_inst.duration,
            )
        ]


class AcquireInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name 'acquire'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        kwargs["name"] = "acquire"
        super().__init__(**kwargs)

    @property
    def pretty_name(self) -> str:
        return self.protocol

    @classmethod
    def list_from_qobj_inst(
        cls,
        qobj_inst: PulseQobjInstruction,
        native_config: NativeQobjConfig,
        hardware_map: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List["AcquireInstruction"]:
        name = qobj_inst.name
        t0 = ceil4(qobj_inst.t0) * 1e-9
        duration = ceil4(qobj_inst.duration) * 1e-9

        return [
            cls(
                name=name,
                t0=t0,
                channel=f"m{qubit_idx}",
                port=hardware_map.get(f"m{qubit_idx}", name),
                duration=duration,
                memory_slot=qobj_inst.memory_slot[n],
                protocol=native_config.protocol.value,
                acq_return_type=native_config.acq_return_type,
                bin_mode=native_config.bin_mode,
            )
            for n, qubit_idx in enumerate(qobj_inst.qubits)
        ]


class DelayInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name 'delay'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        kwargs["name"] = "delay"
        super().__init__(**kwargs)

    @classmethod
    def list_from_qobj_inst(
        cls,
        qobj_inst: PulseQobjInstruction,
        hardware_map: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List["DelayInstruction"]:
        t0 = ceil4(qobj_inst.t0) * 1e-9
        channel = qobj_inst.ch
        duration = ceil4(qobj_inst.duration) * 1e-9

        return [
            cls(
                name=qobj_inst.name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=duration,
            )
        ]


class FreqInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name 'setf'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        assert kwargs["name"] in ("setf",)  # 'shiftf' does not work apparently
        super().__init__(**kwargs)

    @classmethod
    def list_from_qobj_inst(
        cls,
        qobj_inst: PulseQobjInstruction,
        hardware_map: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List["FreqInstruction"]:
        t0 = ceil4(qobj_inst.t0) * 1e-9
        channel = qobj_inst.ch
        frequency = qobj_inst.frequency * 1e9

        return [
            FreqInstruction(
                name=qobj_inst.name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=0.0,
                frequency=frequency,
            )
        ]


class PhaseInstruction(Instruction):
    """Instructions from PulseQobjInstruction with names 'setp', or 'fc'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        assert kwargs["name"] in ("setp", "fc")
        super().__init__(**kwargs)

    @classmethod
    def list_from_qobj_inst(
        cls,
        qobj_inst: PulseQobjInstruction,
        hardware_map: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List["PhaseInstruction"]:
        t0 = ceil4(qobj_inst.t0) * 1e-9
        channel = qobj_inst.ch

        return [
            PhaseInstruction(
                name=qobj_inst.name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=0.0,
                phase=qobj_inst.phase,
            )
        ]


class ParamPulseInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name 'parametric_pulse'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        kwargs["name"] = "parametric_pulse"
        super().__init__(**kwargs)

    @property
    def pretty_name(self) -> str:
        return self.pulse_shape

    @classmethod
    def list_from_qobj_inst(
        cls,
        qobj_inst: PulseQobjInstruction,
        config: PulseQobjConfig,
        hardware_map: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List["ParamPulseInstruction"]:
        t0 = ceil4(qobj_inst.t0) * 1e-9
        channel = qobj_inst.ch
        duration = ceil4(qobj_inst.parameters["duration"]) * 1e-9

        return [
            cls(
                name=qobj_inst.name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=duration,
                pulse_shape=qobj_inst.pulse_shape,
                parameters=qobj_inst.parameters,
            )
        ]


class PulseLibInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name in pulse config library"""

    __slots__ = ()

    @classmethod
    def list_from_qobj_inst(
        cls,
        qobj_inst: PulseQobjInstruction,
        config: PulseQobjConfig,
        native_config: NativeQobjConfig,
        hardware_map: Optional[Dict[str, Any]] = None,
    ) -> List["PulseLibInstruction"]:
        t0 = ceil4(qobj_inst.t0) * 1e-9
        channel = qobj_inst.ch
        name = qobj_inst.name

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

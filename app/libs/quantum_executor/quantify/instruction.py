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

from app.libs.quantum_executor.base.instruction import Instruction


class InitialObjectInstruction(Instruction):
    __slots__ = ()

    def __init__(self, t0=0.0, channel="cl0.baseband", duration=0.0, **kwargs):
        kwargs["name"] = "initial_object"
        super().__init__(t0=t0, channel=channel, duration=duration, **kwargs)


class AcquireInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name 'acquire'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        kwargs["name"] = "acquire"
        super().__init__(**kwargs)

    @property
    def pretty_name(self) -> str:
        return self.protocol


class DelayInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name 'delay'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        kwargs["name"] = "delay"
        super().__init__(**kwargs)


class FreqInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name 'setf'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        assert kwargs["name"] in ("setf",)  # 'shiftf' does not work apparently
        super().__init__(**kwargs)


class PhaseInstruction(Instruction):
    """Instructions from PulseQobjInstruction with names 'setp', or 'fc'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        assert kwargs["name"] in ("setp", "fc")
        super().__init__(**kwargs)


class ParamPulseInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name 'parametric_pulse'"""

    __slots__ = ()

    def __init__(self, **kwargs):
        kwargs["name"] = "parametric_pulse"
        super().__init__(**kwargs)

    @property
    def pretty_name(self) -> str:
        return self.pulse_shape


class PulseLibInstruction(Instruction):
    """Instructions from PulseQobjInstruction with name in pulse config library"""

    __slots__ = ()

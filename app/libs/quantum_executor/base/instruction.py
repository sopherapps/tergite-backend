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
import abc
from typing import List, Optional, Dict, Any
from uuid import uuid4 as uuid

from qiskit.qobj import PulseQobjInstruction, PulseQobjConfig
from quantify_scheduler.enums import BinMode

from app.libs.quantum_executor.base.utils import NativeQobjConfig


class Instruction:
    __slots__ = (
        "t0",
        "name",
        "channel",
        "port",
        "duration",
        "frequency",
        "phase",
        "memory_slot",
        "protocol",
        "parameters",
        "pulse_shape",
        "bin_mode",
        "acq_return_type",
        "label",
    )

    t0: float
    name: str
    channel: str
    port: str
    duration: float
    frequency: float
    phase: float
    memory_slot: List[int]
    protocol: str
    parameters: dict
    pulse_shape: str
    bin_mode: BinMode
    acq_return_type: type
    label: str

    def __init__(self: object, **kwargs):
        self.label = str(uuid())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self: object, other: object) -> bool:
        self_attrs = set(filter(lambda v: hasattr(self, v), Instruction.__slots__))
        other_attrs = set(filter(lambda v: hasattr(other, v), Instruction.__slots__))

        # if they have different attributes, they cannot be equal
        if self_attrs != other_attrs:
            return False

        # label is always unique
        attrs = self_attrs
        attrs.remove("label")

        # if they have the same attributes, they must also all have the same values
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        # otherwise, they are the same
        return True

    def __repr__(self: object) -> str:
        repr_list = [f"Instruction object @ {hex(id(self))}:"]
        for attr in Instruction.__slots__:
            if hasattr(self, attr):
                repr_list.append(f"\t{attr} : {getattr(self, attr)}".expandtabs(4))
        return "\n".join(repr_list)

    @property
    def unique_name(self):
        return f"{self.pretty_name}-{self.channel}-{round(self.t0 * 1e9)}"

    @property
    def pretty_name(self) -> str:
        return self.name

    @classmethod
    @abc.abstractmethod
    def list_from_qobj_inst(
        cls,
        qobj_inst: PulseQobjInstruction,
        config: PulseQobjConfig,
        native_config: NativeQobjConfig,
        hardware_map: Optional[Dict[str, Any]] = None,
    ) -> List["Instruction"]:
        """Generates instances of instruction given a PulseQobjInstruction

        Args:
            qobj_inst: the PulseQobjInstruction to convert from
            config: the PulseQobjConfig for the instruction
            native_config: the native configuration for the qobj
            hardware_map: the mapping of the layout of the physical device

        Returns:
            instances of this class as derived from the qobj_inst
        """
        pass

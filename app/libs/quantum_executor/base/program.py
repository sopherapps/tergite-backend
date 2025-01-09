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

import abc
from dataclasses import dataclass
from typing import FrozenSet

from qiskit.qobj import PulseQobjConfig

from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.utils.logger import ExperimentLogger


@dataclass(frozen=True)
class BaseProgram(abc.ABC):
    name: str
    channels: FrozenSet[Channel]
    config: PulseQobjConfig

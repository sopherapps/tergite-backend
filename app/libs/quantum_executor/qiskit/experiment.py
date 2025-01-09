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

from dataclasses import dataclass

import retworkx as rx

from app.libs.quantum_executor.base.experiment import BaseExperiment
from app.libs.quantum_executor.qiskit.program import QiskitDynamicsProgram

from qiskit.pulse.schedule import Schedule


@dataclass(frozen=True)
class QiskitDynamicsExperiment(BaseExperiment):
    @property
    def schedule(self) -> "Schedule":
        prog = QiskitDynamicsProgram(
            name=self.header.name,
            channels=self.channels,
            config=self.config,
        )
        wccs = rx.weakly_connected_components(self.dag)
        for wcc in wccs:
            wcc_nodes = list(sorted(list(wcc)))
            instruction_types = list(map(lambda i_: self.dag[i_].name, wcc_nodes))
            for i_ in wcc_nodes:
                prog.schedule_operation(
                    self.dag[i_], is_measurement="acquire" in instruction_types
                )
        return prog.schedule

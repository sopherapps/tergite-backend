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

import retworkx as rx
from quantify_scheduler import Schedule

from app.libs.quantum_executor.base.experiment import NativeExperiment
from app.libs.quantum_executor.quantify.program import QuantifyProgram
from app.libs.quantum_executor.utils.general import rot_left
from app.libs.quantum_executor.base.instruction import InitialObjectInstruction

# FIXME: Why is this initial object hard coded here?
initial_object = InitialObjectInstruction()


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

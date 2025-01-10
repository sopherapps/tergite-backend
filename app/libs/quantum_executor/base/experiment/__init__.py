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
#
# Refactored by Martin Ahindura (2024)

import abc
from dataclasses import dataclass
from functools import cached_property
from typing import FrozenSet, List

import retworkx as rx
from pandas import DataFrame
from qiskit.qobj import (
    PulseQobjConfig,
    QobjExperimentHeader,
)

from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.base.instruction import (
    Instruction,
)


@dataclass(frozen=True)
class NativeExperiment(abc.ABC):
    header: QobjExperimentHeader
    instructions: List[Instruction]
    config: PulseQobjConfig
    channels: FrozenSet[Channel]
    buffer_time: float = 0.0

    @cached_property
    def dag(self: "NativeExperiment"):
        dag = rx.PyDiGraph(check_cycle=True, multigraph=False)

        prev_index = dict()
        for j in sorted(self.instructions, key=lambda j: j.t0):
            if j.channel not in prev_index.keys():
                # add the first non-trivial instruction on the channel
                prev_index[j.channel] = dag.add_node(j)
            else:
                # get node index of previous instruction
                i = dag[prev_index[j.channel]]

                # add the next instruction
                prev_index[j.channel] = dag.add_child(
                    parent=prev_index[j.channel], obj=j, edge=j.t0 - (i.t0 + i.duration)
                )

        return dag

    @property
    @abc.abstractmethod
    def schedule(self):
        pass

    @property
    def timing_table(self: "NativeExperiment") -> DataFrame:
        df = self.schedule.timing_table.data
        df.sort_values("abs_time", inplace=True)
        return df

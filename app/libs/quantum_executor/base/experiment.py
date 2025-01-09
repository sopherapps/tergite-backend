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
import copy
from dataclasses import dataclass
from functools import cached_property
from typing import FrozenSet, List, Optional, Dict

import retworkx as rx
from pandas import DataFrame
from qiskit.qobj import PulseQobjConfig, QobjExperimentHeader, PulseQobjExperiment

from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.base.instruction import Instruction, extract_instructions
from app.libs.quantum_executor.utils.general import flatten_list
from app.libs.quantum_executor.utils.logger import ExperimentLogger


@dataclass(frozen=True)
class BaseExperiment(abc.ABC):
    header: QobjExperimentHeader
    instructions: List[Instruction]
    config: PulseQobjConfig
    channels: FrozenSet[Channel]
    buffer_time: float = 0.0

    @cached_property
    def dag(self: "BaseExperiment"):
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
    def timing_table(self: "BaseExperiment") -> DataFrame:
        df = self.schedule.timing_table.data
        df.sort_values("abs_time", inplace=True)
        return df

    @classmethod
    def from_qobj_expt(
        cls,
        expt: PulseQobjExperiment,
        name: str,
        qobj_config: PulseQobjConfig,
        hardware_map: Optional[Dict[str, str]],
    ) -> "BaseExperiment":
        """Converts PulseQobjExperiment to native experiment

        Args:
            expt: the pulse qobject experiment to translate
            name: the name of the experiment
            qobj_config: the pulse qobject config
            hardware_map: the map of the real/simulated device to the logical definitions

        Returns:
            the QiskitDynamicsExperiment corresponding to the PulseQobj
        """
        header = copy.copy(expt.header)
        header.name = name
        inst_nested_list = (
            extract_instructions(
                qobj_inst=inst, config=qobj_config, hardware_map=hardware_map
            )
            for inst in expt.instructions
        )
        native_instructions = flatten_list(inst_nested_list)

        return cls(
            header=header,
            instructions=native_instructions,
            config=qobj_config,
            channels=frozenset(
                Channel(
                    clock=i.channel,
                    frequency=0.0,
                )
                for i in native_instructions
            ),
        )

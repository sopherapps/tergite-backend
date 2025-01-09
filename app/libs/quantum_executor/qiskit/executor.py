# This code is part of Tergite
#
# (C) Stefan Hill (2024)
# (C) Martin Ahindura (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import copy
from functools import partial
from typing import List, Optional, Dict

import numpy as np
import qiskit
import xarray
from qiskit.qobj import PulseQobj, PulseQobjExperiment, PulseQobjConfig
from qiskit.result import Result

import app.libs.storage_file as storagefile
from app.libs.quantum_executor.base.executor import QuantumExecutor
from app.libs.quantum_executor.base.experiment import BaseExperiment
from app.libs.quantum_executor.qiskit.experiment import QiskitDynamicsExperiment
from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.base.instruction import Instruction, extract_instructions
from .backend import QiskitPulse1Q, QiskitPulse2Q
from .transpile import transpile
from ..utils.general import flatten_list
from ...properties import BackendConfig


class QiskitDynamicsExecutor(QuantumExecutor):
    def __init__(self, backend_config: BackendConfig):
        super().__init__(experiment_cls=QiskitDynamicsExperiment)
        self.backend = QiskitPulse1Q(backend_config=backend_config)

    def run(self, experiment: BaseExperiment, /) -> xarray.Dataset:
        job = self.backend.run(experiment.schedule)
        result: Result = job.result()
        return result.data()["memory"]

    def close(self):
        pass


class QiskitPulse1QExecutor(QuantumExecutor):
    def __init__(self, backend_config: BackendConfig):
        super().__init__(experiment_cls=QiskitDynamicsExperiment)
        # TODO: Use measurement level provided by the client request if discriminator is not provided
        self.backend = QiskitPulse1Q(
            meas_level=1, meas_return="single", backend_config=backend_config
        )
        self.shots = 1024

    def run(self, experiment: BaseExperiment, /) -> xarray.Dataset:
        job = self.backend.run(
            experiment, shots=self.shots, meas_return=self.meas_return
        )
        result = job.result()
        data = result.data()["memory"]

        # Combine real and imaginary parts into complex numbers
        if self.meas_return == "avg":
            # for meas_return avg, there is only one data point averaged across the shots
            # Create acquisition index coordinate that matches the length of complex_data
            acq_index = np.arange(
                data.shape[0]
            )  # Should match the number of rows in complex_data
            complex_data = data[:, 0] + 1j * data[:, 1]
        else:
            # Create acquisition index coordinate that matches the length of complex_data
            acq_index = np.arange(
                data.shape[1]
            )  # Should match the number of rows in complex_data

            complex_data = data[:, 0, 0] + 1j * data[:, 0, 1]

        coords = {
            "acq_index_0": acq_index,  # Coordinate array that matches the dimension length
        }

        # Create the xarray Dataset
        ds = xarray.Dataset(
            data_vars={
                "0": (
                    ["repetition", "acq_index_0"],
                    np.expand_dims(complex_data, axis=1),
                )
            },
            coords=coords,
        )

        return ds

    # def construct_experiments(self, qobj: PulseQobj, /) -> List[QiskitDynamicsExperiment]:
    #     # because we avoid experiments structure we have to pass shots and
    #     # measurement level configurations to the run function
    #     self.shots = qobj.config.shots
    #     self.meas_return = qobj.config.meas_return
    #     qobj_dict = qobj.to_dict()
    #     tx = transpile(qobj_dict)
    #
    #     self.logger.info(f"Translated {len(tx)} OpenPulse experiments.")
    #     return tx

    def close(self):
        pass


class QiskitPulse2QExecutor(QuantumExecutor):
    def __init__(self, backend_config: BackendConfig):
        super().__init__()
        self.backend = QiskitPulse2Q(
            meas_level=1, meas_return="single", backend_config=backend_config
        )

    def run(self, experiment: BaseExperiment, /) -> xarray.Dataset:
        job = self.backend.run(
            experiment, shots=self.shots, meas_return=self.meas_return
        )
        result = job.result()
        data = result.data()["memory"]

        num_measured = data.shape[1]
        if self.meas_return == "avg":
            raise NotImplementedError("Not implemented 2q avg.")
        else:
            # TODO: depending on the measurement level, adjust dataset structure
            # Combine real and imaginary parts into complex numbers
            complex_data = data[:, :, 0] + 1j * data[:, :, 1]

        # Create acquisition index coordinate that matches the length of a single repetition
        acq_index = np.arange(1)

        coords = {}
        data_vars = {}
        for i in range(num_measured):
            coords[f"acq_index_{i}"] = acq_index
            data_vars[f"{i}"] = (
                ["repetition", f"acq_index_{i}"],
                np.expand_dims(complex_data[:, i], axis=1),
            )

        ds = xarray.Dataset(data_vars=data_vars, coords=coords)
        return ds

    # def construct_experiments(self, qobj: PulseQobj, /) -> List[QiskitDynamicsExperiment]:
    #     qobj_dict = qobj.to_dict()
    #     self.shots = qobj.config.shots
    #     self.meas_return = qobj.config.meas_return
    #     tx = transpile(qobj_dict)
    #
    #     self.logger.info(f"Translated {len(tx)} OpenPulse experiments.")
    #     return tx

    def close(self):
        pass

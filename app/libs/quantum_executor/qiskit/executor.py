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
import abc
from typing import Type, List

import numpy as np
import xarray
from qiskit.qobj import PulseQobj

from app.libs.quantum_executor.base.executor import QuantumExecutor
from app.libs.quantum_executor.base.experiment import NativeExperiment
from app.libs.quantum_executor.qiskit.experiment import QiskitDynamicsExperiment
from .backends.two_qubit import QiskitPulse2Q
from .backends.one_qubit import QiskitPulse1Q
from .backends.base import QiskitPulseBackend
from ..base.utils import MeasRet, NativeQobjConfig
from ..utils.general import get_experiment_name
from ..utils.logger import ExperimentLogger
from ...properties import BackendConfig


class QiskitDynamicsExecutor(QuantumExecutor, abc.ABC):
    def __init__(
        self,
        backend_config: BackendConfig,
        backend_cls: Type[QiskitPulseBackend] = QiskitPulse1Q,
        **kwargs,
    ):
        """Creates the QiskitDynamicsExecutor for the given backend class and key-word args

        Args:
            backend_config: the configuration of the backend
            backend_cls: the class of the backend
            kwargs: extra key-word args to pass to the backend on initialisation
        """
        super().__init__()
        self.backend = backend_cls(backend_config=backend_config, **kwargs)

    def _to_native_experiments(
        self, qobj: PulseQobj, native_config: NativeQobjConfig, /
    ) -> List[QiskitDynamicsExperiment]:
        """Constructs qiskit dynamics experiments from the PulseQobj instance

        Args:
            qobj: the Pulse qobject containing the experiments
            native_config: the native config for the qobj

        Returns:
            list of QiskitDynamicsExperiment's
        """
        native_experiments = [
            QiskitDynamicsExperiment.from_qobj_expt(
                name=get_experiment_name(expt.header.name, idx + 1),
                expt=expt,
                qobj_config=qobj.config,
            )
            for idx, expt in enumerate(qobj.experiments)
        ]
        return native_experiments

    def close(self):
        pass


class QiskitPulse1QExecutor(QiskitDynamicsExecutor):
    def __init__(self, backend_config: BackendConfig):
        # TODO: Use measurement level provided by the client request if discriminator is not provided
        super().__init__(
            backend_config=backend_config,
            backend_cls=QiskitPulse1Q,
            meas_level=1,
            meas_return="single",
        )

    def _run_native(
        self,
        experiment: NativeExperiment,
        /,
        *,
        native_config: NativeQobjConfig,
        logger: ExperimentLogger,
    ) -> xarray.Dataset:
        meas_return = native_config.meas_return
        shots = native_config.shots
        job = self.backend.run(
            experiment.schedule, shots=shots, meas_return=meas_return.value
        )
        result = job.result()
        data = result.data()["memory"]

        # Combine real and imaginary parts into complex numbers
        if meas_return == MeasRet.AVERAGED:
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


class QiskitPulse2QExecutor(QiskitDynamicsExecutor):
    def __init__(self, backend_config: BackendConfig):
        super().__init__(
            backend_config=backend_config,
            backend_cls=QiskitPulse2Q,
            meas_level=1,
            meas_return="single",
        )

    def _run_native(
        self,
        experiment: NativeExperiment,
        /,
        *,
        native_config: NativeQobjConfig,
        logger: ExperimentLogger,
    ) -> xarray.Dataset:
        meas_return = native_config.meas_return
        shots = native_config.shots
        job = self.backend.run(
            experiment.schedule, shots=shots, meas_return=meas_return.value
        )
        result = job.result()
        data = result.data()["memory"]

        num_measured = data.shape[1]
        if meas_return == MeasRet.AVERAGED:
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

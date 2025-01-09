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
import copy
import json
from datetime import datetime
from pathlib import Path
from traceback import format_exc
from typing import Optional, List, Type, Dict

import numpy as np
import rich
import xarray
from qiskit.providers.ibmq.utils.json_encoder import IQXJsonEncoder as PulseQobj_encoder
from qiskit.qobj import PulseQobj
from quantify_core.data import handling as dh
from quantify_core.data.handling import create_exp_folder, gen_tuid

import settings
from app.libs.quantum_executor.base.experiment import BaseExperiment
from app.libs.quantum_executor.base.instruction import get_meas_settings
from app.libs.quantum_executor.utils.logger import ExperimentLogger
from app.libs.storage_file.file import QuantumJob, StorageFile


class QuantumExecutor(abc.ABC):
    def __init__(
        self,
        experiment_cls: Type[BaseExperiment],
        hardware_map: Optional[Dict[str, str]] = None,
    ):
        dh.set_datadir(settings.EXECUTOR_DATA_DIR)
        self.experiment_cls: Type[BaseExperiment] = experiment_cls
        self.hardware_map = hardware_map

    def register_job(self, tag: str = ""):
        # TODO: The fields tuid, experiment_folder, and logger could be class properties
        self.tuid = gen_tuid()
        self.experiment_folder = Path(create_exp_folder(tuid=self.tuid, name=tag))
        self.logger = ExperimentLogger(self.tuid)
        self.logger.info(f"Registered job: {self.tuid}")

    def debug_save_qobj(self, qobj: PulseQobj):
        """Saves the incoming PulseQobj for debugging.
        This is a re-encoding when using the rest_api, but it is needed for local debugging.
        TODO: Avoid re-encoding for external jobs.
        """
        file = self.experiment_folder / "qobj.json"
        with open(file, mode="w") as qj:
            json.dump(qobj.to_dict(), qj, cls=PulseQobj_encoder, indent="\t")
        rich.print(f"Saved PulseQobj at {file}")
        self.logger.info(f"Saved PulseQobj at {file}")

    def construct_experiments(self, qobj: PulseQobj, /) -> List[BaseExperiment]:
        """Constructs native experiments from the PulseQobj instance

        Args:
            qobj: the Pulse qobject containing the experiments

        Returns:
            list of QuantifyExperiment's
        """
        self.logger.info(f"Compiling qobj")
        native_experiments = [
            self.experiment_cls.from_qobj_expt(
                name=StorageFile.sanitized_name(expt.header.name, idx + 1),
                expt=expt,
                qobj_config=qobj.config,
                hardware_map=self.hardware_map,
            )
            for idx, expt in enumerate(qobj.experiments)
        ]
        self.logger.info(f"Translated {len(native_experiments)} OpenPulse experiments.")
        return native_experiments

    @abc.abstractmethod
    def run(self, experiment: BaseExperiment, /) -> xarray.Dataset:
        pass

    def run_experiments(
        self,
        qobj: PulseQobj,
        /,
        *,
        job_id: str = None,
    ) -> Optional[Path]:
        """Runs the experiments and returns the results file path

        Args:
            qobj: the Quantum object that is to be executed
            job_id: the ID of the job

        Returns:
            the path to the results obtained after measurement
        """
        self.debug_save_qobj(qobj)
        try:
            # unwrap pulse library
            qobj.config.pulse_library = {
                i.name: np.asarray(i.samples) for i in qobj.config.pulse_library
            }

            # translate qobj experiments to quantify schedules
            self.logger.info(
                f"Starting constructing experiments for job id: {job_id} at {datetime.now()}"
            )
            tx = self.construct_experiments(qobj)

            program_settings = get_meas_settings(qobj.config)
            for k, v in program_settings.dict().items():
                self.logger.info(f"Set {k} to {v}")

            self.logger.info(f"Running experiments for job id: {job_id}")
            experiment_results = {
                expt.header.name: self.run(expt).to_dict() for expt in tx
            }

            job = QuantumJob(
                job_id=job_id,
                tuid=self.tuid,
                meas_return=program_settings.meas_return,
                meas_return_cols=program_settings.meas_return_cols,
                meas_level=program_settings.meas_level,
                memory_slot_size=qobj.config.memory_slot_size,
                qobj=qobj,
                header=qobj.header,
                experiments=experiment_results,
            )

            filename = "measurement.hdf5" if job_id is None else f"{job_id}.hdf5"
            results_file_path = self.experiment_folder / filename
            job.to_hdf5(results_file_path)

            self.logger.info(f"Stored measurement data at {results_file_path}")
            self.logger.info(
                f"Completed {job_id if job_id else 'local job'} with tuid {self.tuid}."
            )

        # record exceptions
        except Exception as e:
            self.logger.error(
                f"\nFailed job: {job_id}, tuid: {self.tuid}\n{format_exc()}"
            )
            raise e

        return results_file_path

    @abc.abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

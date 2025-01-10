# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright David Wahlstedt 2021, 2022, 2023
# (C) Copyright Abdullah Al Amin 2021, 2022
# (C) Copyright Axel Andersson 2022
# (C) Andreas Bengtsson 2020
# (C) Martin Ahindura 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import functools
import logging
import shutil
from pathlib import Path
from typing import Any, Tuple, Type

import numpy as np
import numpy.typing as npt
import redis
import requests
import rq.job
from requests import Response
from sklearn.utils.extmath import safe_sparse_dot

import settings
from app.libs.quantum_executor.base.job import StorageFile
from app.libs.quantum_executor.base.utils import MeasLvl
from app.services.jobs.workers.postprocessing.exc import PostProcessingError
from app.libs.properties.utils import date_time
from app.utils.http import get_mss_client

from ...service import (
    Location,
    fetch_job,
    fetch_redis_entry,
    inform_location,
    save_result,
    update_final_location_timestamp,
)

# Storage settings

STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
LOGFILE_DOWNLOAD_POOL_DIRNAME = settings.LOGFILE_DOWNLOAD_POOL_DIRNAME

# Connectivity settings

MSS_MACHINE_ROOT_URL = settings.MSS_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL

LOCALHOST = "localhost"

# REST API

REST_API_MAP = {
    "timelog": "/timelog",
    "jobs": "/jobs",
    "logfiles": "/logfiles",
    "backends": "/backends",
}

# Type aliases

JobID = str


# =========================================================================
# Post-processing entry function
# =========================================================================


def logfile_postprocess(logfile: Path) -> JobID:
    print(f"Postprocessing logfile {str(logfile)}")

    # Move the logfile to logfile download pool area
    # TODO: This file change should preferably happen _after_ the
    new_file_name = Path(logfile).stem  # This is the job_id
    new_file_name_with_suffix = new_file_name + ".hdf5"
    storage_location = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME

    new_file_path = storage_location / LOGFILE_DOWNLOAD_POOL_DIRNAME
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name_with_suffix

    shutil.move(logfile, new_file)

    print(f"Moved the logfile to {str(new_file)}")

    # Inform job supervisor
    inform_location(new_file_name, Location.PST_PROC_W)

    # The return value will be passed to postprocessing_success_callback
    print("Identified TQC storage file, reading file using storage file")
    sf = StorageFile(new_file, mode="r")
    return postprocess_storage_file(sf)


# =========================================================================
# Post-processing Quantify / Qblox files
# =========================================================================


def _apply_linear_discriminator(
    backend: dict, qubit_idx: int, iq_points: npt.NDArray[np.complex128]
) -> npt.NDArray[np.int_]:
    """
    Fetches the linear discriminator from the backend definition

    Args:
        backend: Backend definition as dictionary
        qubit_idx: ID of the qubit to discriminate
        iq_points: IQ points from the measurement

    Returns:
        Discriminated 0 and 1 states as numpy array

    """
    discriminator_ = backend["discriminators"]["lda"]
    # TODO: We are having two "qubit_id" (e.g. q12 = 0, q13 = 1) and we should have some more meaningful representation
    qubit_id_ = backend["qubit_ids"][qubit_idx]
    coefficients = np.array(
        [
            discriminator_[qubit_id_]["coef_0"],
            discriminator_[qubit_id_]["coef_1"],
        ]
    )
    intercept = np.array(discriminator_[qubit_id_]["intercept"])

    X = np.zeros((iq_points.shape[0], 2))
    X[:, 0] = iq_points.real
    X[:, 1] = iq_points.imag

    scores = safe_sparse_dot(X, coefficients.T, dense_output=True) + intercept

    return (scores.ravel() > 0).astype(np.int_)


def postprocess_storage_file(
    sf: StorageFile, backend_name: str = settings.DEFAULT_PREFIX
) -> JobID:
    try:
        with get_mss_client() as mss_client:
            if sf.meas_level == MeasLvl.DISCRIMINATED:
                # This would fetch the discriminator from the MSS
                backend_definition: str = f'{str(MSS_MACHINE_ROOT_URL)}{REST_API_MAP["backends"]}/{backend_name}'
                response = mss_client.get(backend_definition)

                if response.status_code == 200:
                    discriminator_fn = functools.partial(
                        _apply_linear_discriminator, response.json()
                    )
                else:
                    print(f"Response error {response}")

                try:
                    memory = sf.as_readout(discriminator=discriminator_fn)
                    save_result_in_mss_and_bcc(
                        mss_client=mss_client, memory=memory, job_id=sf.job_id
                    )
                except Exception as exp:
                    logging.error(exp)
                    response = _update_job_in_mss(
                        mss_client=mss_client,
                        job_id=sf.job_id,
                        payload={"status": "ERROR"},
                    )
                    raise exp

            elif sf.meas_level == MeasLvl.INTEGRATED:
                memory = sf.as_xarray()

                save_result_in_mss_and_bcc(
                    mss_client=mss_client, memory=memory, job_id=sf.job_id
                )

            elif sf.meas_level == MeasLvl.RAW:
                # TODO: Not implemented, expects to pass full trace
                # with hardcoded 16K values length array
                save_result_in_mss_and_bcc(
                    mss_client=mss_client, memory=[], job_id=sf.job_id
                )

            else:
                print("Warning: cannot postprocess invalid StorageFile.")

            # job["name"] was set to "pulse_schedule" when registered

        return sf.job_id
    except Exception as exp:
        raise PostProcessingError(exp=exp, job_id=sf.job_id)


# =========================================================================
# Post-processing success callback with helper
# =========================================================================


def postprocessing_success_callback(
    _rq_job, _rq_connection, result: JobID, *args, **kwargs
):
    # From logfile_postprocess:
    job_id = result
    inform_location(job_id, Location.FINAL_Q)
    update_final_location_timestamp(job_id, status="started")

    script_name, post_processing = get_metainfo(job_id)

    status = fetch_job(job_id, "status")

    with get_mss_client() as mss_client:
        if status["failed"]["time"]:
            print(
                f"Job {job_id}, {script_name=}, {post_processing=} has failed: aborting. Status: {status}"
            )
            _update_location_timestamps_in_mss(mss_client=mss_client, job_id=job_id)
            return

        print(f"Job with ID {job_id}, {script_name=} has finished")
        if post_processing:
            print(
                f"Results post-processed by '{post_processing}' available by job_id in Redis."
            )

        update_final_location_timestamp(job_id, status="finished")
        _update_location_timestamps_in_mss(mss_client=mss_client, job_id=job_id)
        inform_location(job_id, Location.FINAL_W)


# job, connection, type, value, traceback
def postprocessing_failure_callback(
    _rq_job: rq.job.Job,
    _rq_connection: redis.Redis,
    _type: Type,
    value: Any,
    traceback: Any,
):
    """Callback to be called when postprocessing fails"""
    with get_mss_client() as mss_client:
        if isinstance(value, PostProcessingError):
            _update_location_timestamps_in_mss(
                mss_client=mss_client, job_id=value.job_id
            )


def get_metainfo(job_id: str) -> Tuple[str, str]:
    entry = fetch_redis_entry(job_id)
    script_name = entry["name"]
    post_processing = entry.get("post_processing")
    return script_name, post_processing


# =========================================================================
# BCC / MSS updating
# =========================================================================


def save_result_in_mss_and_bcc(mss_client: requests.Session, memory, job_id: JobID):
    """Updates both MSS and BCC with the memory part of the result

    Args:
        mss_client: the requests.Session that can query MSS
        memory: the memory part of the result, usually saved as {'result': {'memory': [...]}}
        job_id: the ID of the job
    """
    _debug_job_memory_list(memory)
    result = {"memory": memory}
    payload = {
        "result": result,
        "status": "DONE",
        "download_url": f'{BCC_MACHINE_ROOT_URL}{REST_API_MAP["logfiles"]}/{job_id}',
        "timelog.RESULT": date_time.utc_now_iso(),
    }

    # update BCC's redis database
    save_result(job_id, result)

    # update MSS
    response = _update_job_in_mss(mss_client=mss_client, job_id=job_id, payload=payload)
    if response:
        print("Pushed update to MSS")


def _update_location_timestamps_in_mss(mss_client: requests.Session, job_id: JobID):
    """Updates the job entry in MSS with the timestamps that have been saved for each pipeline location

    Args:
        mss_client: the requests.Session that can query MSS
        job_id: the ID of the job
    """
    entry = fetch_redis_entry(job_id)
    try:
        payload = {"timestamps": entry["timestamps"]}
        response = _update_job_in_mss(
            mss_client=mss_client, job_id=job_id, payload=payload
        )
        if not response.ok:
            raise ValueError(
                f"failed to push timestamps for job {job_id}", response.text
            )
    except Exception as exp:
        logging.error(exp)
        raise exp


def _update_job_in_mss(
    mss_client: requests.Session, job_id: JobID, payload: dict
) -> Response:
    """Updates the job in MSS with the given payload

    Args:
        mss_client: the requests.Session that can query MSS
        job_id: the ID of the job
        payload: the new updates to apply to the given job in MSS

    Returns:
        the requests.Response received after request to MSS
    """
    mss_job_url = f'{MSS_MACHINE_ROOT_URL}{REST_API_MAP["jobs"]}/{job_id}'
    return mss_client.put(mss_job_url, json=payload)


def _debug_job_memory_list(memory: list):
    """
    Helper printout with first 5 outcomes
    """
    print("Measurement results:")
    for experiment_memory in memory:
        s = str(experiment_memory[:5])
        if experiment_memory[5:6]:
            s = s.replace("]", ", ...]")
        print(s)

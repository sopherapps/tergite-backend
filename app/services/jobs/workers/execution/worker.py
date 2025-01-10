# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright Abdullah-Al Amin 2021
# (C) Copyright Axel Andersson 2022
# (C) Copyright David Wahlstedt 2022
# (C) Copyright Martin Ahindura 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import json
from datetime import datetime
from pathlib import Path

from qiskit.qobj import PulseQobj
from qiskit_ibm_provider.utils import json_decoder
from redis import Redis

import settings
from app.libs.quantum_executor.utils.connections import get_executor_lock
from app.libs.quantum_executor.utils.serialization import iqx_rld
from app.utils.queues import QueuePool
from .utils import get_executor

from ...service import Location, fetch_job, inform_failure, inform_location
from ..postprocessing import (
    logfile_postprocess,
    postprocessing_failure_callback,
    postprocessing_success_callback,
)

# Settings
# --------
STORAGE_ROOT = settings.STORAGE_ROOT
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
DEFAULT_PREFIX = settings.DEFAULT_PREFIX


# Redis connection
# ----------------
rq_queues = QueuePool(prefix=DEFAULT_PREFIX, connection=settings.REDIS_CONNECTION)
executor = get_executor()


def job_execute(job_file: Path):
    print(f"Executing file {str(job_file)}")

    with job_file.open() as f:
        job_dict = json.load(f)

    job_id = ""
    try:
        job_id = job_dict["job_id"]

        # Inform supervisor
        inform_location(job_id, Location.EXEC_W)

        qobj = job_dict["params"]["qobj"]

        # --- RLD pulse library
        # [([a,b], 2),...] -> [[a,b],[a,b],...]
        for pulse in qobj["config"]["pulse_library"]:
            pulse["samples"] = iqx_rld(pulse["samples"])

    except KeyError as exp:
        print("Invalid job")
        print(f"Job execution failed. Key error: {exp}")
        inform_failure(job_id, reason="malformed job")
        return {"message": "malformed job"}

    # Just a locking mechanism to ensure jobs don't interfere with each other
    with get_executor_lock():
        try:
            # --- In-place decode complex values
            # [[a,b],[c,d],...] -> [a + ib,c + id,...]
            json_decoder.decode_pulse_qobj(qobj)

            print(datetime.now(), "IN REST API CALLING RUN_EXPERIMENTS")

            results_file = executor.run(PulseQobj.from_dict(qobj), job_id=job_id)
        except Exception as exp:
            print("Job failed")
            print(f"Job execution failed. exp: {exp}")
            # inform supervisor about failure
            inform_failure(job_id, reason="no response")
            return {"message": "failed"}

    if results_file:
        job_status = fetch_job(job_id, "status")
        if job_status["cancelled"]["time"]:
            print("Job cancelled, postprocessing halted")
            # FIXME: Probably provide an error message to the client also
            return {"message": "cancelled"}

        rq_queues.logfile_postprocessing_queue.enqueue(
            logfile_postprocess,
            on_success=postprocessing_success_callback,
            on_failure=postprocessing_failure_callback,
            job_id=f"{job_id}_{Location.PST_PROC_Q.name}",
            args=(results_file,),
        )

        # inform supervisor
        inform_location(job_id, Location.PST_PROC_Q)

    # clean up
    job_file.unlink(missing_ok=True)
    print("Job executed successfully")
    return {"message": "ok"}

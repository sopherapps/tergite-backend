# This code is part of Tergite
#
# (C) Copyright Nicklas Botö, Fabian Forslund 2022
# (C) Copyright David Wahlstedt 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from pathlib import Path

from redis import Redis

import settings

from ....utils.json import get_items_from_json
from ....utils.queues import QueuePool
from ..service import Location, inform_location, register_job, update_job_entry
from .preprocessing import job_preprocess

# settings
DEFAULT_PREFIX = settings.DEFAULT_PREFIX
STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
JOB_EXECUTION_POOL_DIRNAME = settings.JOB_EXECUTION_POOL_DIRNAME
JOB_PRE_PROC_POOL_DIRNAME = settings.JOB_PRE_PROC_POOL_DIRNAME


# TODO: Get rid of redis-based rq queueing. Let a simple FIFO queue be implemented using
#     timestamped sqlite entries. Take note the sqlite does not play well in multithreaded in the default setting.
#     Reasons for change:
#        - fewer dependencies, sqlite comes pre-installed in python
#        - higher speed of access since there is no network call. SQLite is embedded
#        - simpler code, flowing as a single process. 
#          Gotcha: While execution is mostly IO and can be made asynchronous, state discrimination maybe CPU-intensive.
#          It is expected that some form of multiprocessing must happen.


# preprocessing queue
rq_queues = QueuePool(prefix=DEFAULT_PREFIX, connection=settings.REDIS_CONNECTION)


def job_register(job_file: Path) -> None:
    """Registers job in job supervisor"""
    job_id = job_file.stem
    # inform job supervisor about job registration
    print(f"Registering job file {str(job_file)}")
    register_job(job_id)
    # store the received file in the job upload pool
    new_file_name = job_file.stem
    new_file_path = (
        Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME / JOB_PRE_PROC_POOL_DIRNAME
    )
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name
    job_file.replace(new_file)
    # add job to pre-processing queue and notify job supervisor
    rq_queues.job_preprocessing_queue.enqueue(
        job_preprocess,
        new_file,
        job_id=job_id + f"_{Location.PRE_PROC_Q.name}",
    )
    inform_location(job_id, Location.PRE_PROC_Q)

    # put some of this job's items in job_supervisor's Redis entry
    keys = ["name", "is_calibration_supervisor_job", "post_processing"]
    dict_partial = get_items_from_json(new_file, keys)
    for key, value in dict_partial.items():
        update_job_entry(job_id, value, key)

# This code is part of Tergite
#
# (C) Copyright Axel Andersson 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import pathlib
from os.path import exists

import h5py

from app.libs.quantum_executor.base.job import StorageFile
from app.libs.quantum_executor.base.utils import MeasLvl, MeasRet

# ensure data directory exists
pathlib.Path("pytest-data").mkdir(parents=True, exist_ok=True)


def test_rw():
    fn = pathlib.Path("pytest-data/pytest-rw.hdf5")
    sf = StorageFile(
        fn,
        mode="w",
        tuid="pytest",
        meas_return=MeasRet.APPENDED,
        meas_level=MeasLvl.INTEGRATED,
        meas_return_cols=1,
        job_id="pytest",
    )
    assert exists(fn), f"File '{fn}' does not exist after writing."
    sf.file.close()
    sf = StorageFile(fn, mode="r")
    for attr, attr_type in {
        "job_id": str,
        "tuid": str,
        "local": bool,
        "meas_level": MeasLvl,
        "meas_return": MeasRet,
        "header": h5py.Group,
        "experiments": h5py.Group,
    }.items():
        assert hasattr(sf, attr), f"StorageFile does not have '{attr}' attribute."
        v = getattr(sf, attr)
        assert isinstance(
            v, attr_type
        ), f"Attribute '{attr}' should be type {attr_type} but is type {type(v)}."

    sf.file.close()
    os.remove(fn)
    assert not exists(fn), f"File '{fn}' was not removed after deletion."

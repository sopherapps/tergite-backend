# This code is part of Tergite
#
# (C) Copyright Axel Andersson 2022
# (C) Copyright Martin Ahindura 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import functools
import json
from enum import Enum
from pathlib import Path
from typing import Literal, Union, Optional, Any, Dict, List, Tuple, TypedDict

import h5py
import numpy as np
import pydantic
import xarray as xr

from qiskit.providers.ibmq.utils.json_encoder import IQXJsonEncoder as PulseQobj_encoder
from qiskit.qobj import PulseQobj, QobjHeader

from . import utils as parse

JOB_HDF5_FILE_DELIMITER = "~"


class MeasLvl(int, Enum):
    DISCRIMINATED = 2
    INTEGRATED = 1
    RAW = 0


class MeasRet(int, Enum):
    AVERAGED = 1
    APPENDED = 0


class RegisterOrder(str, Enum):
    LITTLE_ENDIAN = "little"
    BIG_ENDIAN = "big"


class RegisterSparsity(str, Enum):
    FULL = "full"
    SPARSE = "sparse"


class StorageFile:
    delimiter = "~"

    def __init__(
        self,
        # filepath in OS
        storage_file: Path,
        /,
        *,
        # mode specifier
        mode: Literal["r", "w"],
        # during write mode, the following need to be specified
        tuid: Union[str, None] = None,
        meas_return: Union[MeasRet, None] = None,
        meas_level: Union[MeasLvl, None] = None,
        meas_return_cols: Union[int, None] = None,
        # during write mode, the following are optional
        job_id: Union[str, None] = None,
        memory_slot_size: int = 100,  # the maximum number of classical register slots
    ):
        pass

    # TODO: leave register_sparsity as full and set it to sparse for cases where there are multiple values per shot
    def as_readout(
        self: "StorageFile",
        discriminator: callable,
        *,
        disc_two_state: bool = False,
        register_order: str = "little",
        register_sparsity: str = "full",
    ) -> list:
        """
        Interpret measurement data from experiments as bitstrings.
        Requires a callable discriminator function.

        The discriminator has to be a Python function
        which takes two arguments "qubit_index" and "iq_point"
        and returns a binary value (0/1).

        Returns an object of type List[List[hex]]
        where the outer dimension is the experiment number.
        The inner dimension is the shot number.

        The hex value corresponds to how the classical slot register
        is read "from right to left" (Little endian) in binary. This
        binary value is then converted to hex (base 16).
        """

        # ------------ Helper functions
        def _register_full(register: list, *, slot_idxs: list) -> str:
            full_register = ["0" for _ in range(self.memory_slot_size)]
            for i, idx in enumerate(slot_idxs):
                full_register[self.memory_slot_size - 1 - idx] = str(register[i])
            return "".join(full_register)

        def _register_sparse(register: list, **kwargs) -> str:
            return "".join(register)

        def _map_to_hex(
            binary_transposed_msmt_matx, *, slot_idxs: list, register_parse_fn: callable
        ) -> list:
            le_registers = map(
                functools.partial(register_parse_fn, slot_idxs=slot_idxs),
                binary_transposed_msmt_matx.astype(str),
            )
            b_int = 3 if disc_two_state else 2
            hex_le_registers = map(lambda reg: hex(int(reg, b_int)), le_registers)
            return list(hex_le_registers)

        # ------------ Normalize arguments
        register_sparsity = str.strip(str.lower(register_sparsity))
        register_order = str.strip(str.lower(register_order))

        # ------------ Select if register readout should be sparse or not
        if RegisterSparsity(register_sparsity) == RegisterSparsity.SPARSE:
            register_parse_fn = _register_sparse
        elif RegisterSparsity(register_sparsity) == RegisterSparsity.FULL:
            register_parse_fn = _register_full
        else:
            raise ValueError(f"Invalid register sparsity setting: {register_sparsity}.")

        # ------------ Select register bit reading order
        if RegisterOrder(register_order) == RegisterOrder.LITTLE_ENDIAN:
            register_reverse = True
        elif RegisterOrder(register_order) == RegisterOrder.BIG_ENDIAN:
            register_reverse = False
        else:
            raise ValueError(f"Invalid register order setting: {register_order}.")

        # ------------ Discriminate the I/Q readout
        assert callable(discriminator)
        readout = list()
        for tag, experiment_data in self.sort_items(self.experiments.items()):
            memory = list()

            slots = filter(lambda item: "slot" in item[0], experiment_data.items())

            # keep track of which classical register slots were used
            # these will be in little endian order
            slot_idxs = list()

            # sort in reverse for little endian (descending slot order)
            for slot_tag, slot_data in self.sort_items(slots, reverse=register_reverse):
                # TODO: assert if measurement shape if equal to the number of shots
                # assert (
                #     slot_data["measurement"].shape[0] == 1
                # ), "Max one acquisition per channel for word readout."
                kets = np.zeros(slot_data["measurement"].shape[0]).astype(int)
                slot_idx = int(slot_tag.split(self.delimiter)[1])
                slot_idxs.append(slot_idx)
                row = slot_data["measurement"][0, :]
                kets = discriminator(qubit_idx=slot_idx, iq_points=row)
                memory.append(kets)
            # binary matrix where rows are classical register values
            # and columns are register value per shot
            memory = np.asarray(memory)
            # transposing the memory we get a binary matrix where columns are
            # classical register values and rows are value per shot
            memory = np.transpose(memory)
            # get hexlist of classical registers for every shot
            # and append to readout list
            readout.append(
                _map_to_hex(
                    memory, slot_idxs=slot_idxs, register_parse_fn=register_parse_fn
                )
            )

        return readout

    def as_xarray(self: "StorageFile") -> xr.Dataset:
        """Attempts to parse the storage file as an N-dimensional parametric sweep.
        Returns an xarray dataset. Automatically detects multiplexed sweeps.

        Assumes that the sweep order of the independent variables is in the order
        of main_dims, e.g.
            if main_dims = ("frequencies", "amplitudes"),
            then assumes index order is:
                for f in frequencies:
                    for a in amplitudes:
                        ...
        """
        if (self.meas_return == MeasRet.APPENDED) and (
            self.meas_level == MeasLvl.DISCRIMINATED
        ):
            return NotImplemented  # discriminator?

        elif (self.meas_return == MeasRet.AVERAGED) and (
            self.meas_level == MeasLvl.DISCRIMINATED
        ):
            return NotImplemented  # discriminator?

        elif (self.meas_return == MeasRet.APPENDED) and (
            self.meas_level == MeasLvl.INTEGRATED
        ):
            return parse.appended_integrated(data=self)

        elif (self.meas_return == MeasRet.AVERAGED) and (
            self.meas_level == MeasLvl.INTEGRATED
        ):
            return parse.averaged_integrated(data=self)

        elif (self.meas_return == MeasRet.AVERAGED) and (
            self.meas_level == MeasLvl.RAW
        ):
            return parse.averaged_raw(data=self)

        else:
            raise NotImplementedError(
                f"Invalid storage file metadata: {self.meas_return} and {self.meas_level} is not implemented."
            )

    @functools.cached_property
    def sorted_measurements(self: "StorageFile") -> list:
        # TODO: this would only find files for simulated readout output
        return sorted(
            parse.find(self.experiments, "measurement"),
            key=lambda path: path[0].split(self.delimiter)[1],
        )

    @classmethod
    def sort_items(cls: "StorageFile", items: list, reverse: bool = False) -> iter:
        return sorted(
            items, key=lambda tup: int(tup[0].split(cls.delimiter)[1]), reverse=reverse
        )

    @staticmethod
    def sanitized_name(users_experiment_name: str, experiment_index: int):
        """Return a cleaned version of a given experiment name."""
        name = "".join(
            x for x in users_experiment_name if x.isalnum() or x in " -_,.()"
        )
        return f"{name}{StorageFile.delimiter}{experiment_index}"


class QobjMetadata(pydantic.BaseModel):
    """Metadata on a Qobject instance"""

    shots: int
    qobj_id: str
    num_experiments: int

    @classmethod
    def from_qobj(cls, qobj: PulseQobj):
        """Constructs the metadata from the qobject
        Args:
            qobj: the qobject whose metadata is to be obtained

        Returns:
            the QobjectMetadata for the given qobject
        """
        return cls(
            shots=qobj.config.shots,
            qobj_id=qobj.qobj_id,
            num_experiments=len(qobj.experiments),
        )


class XArrayDict(TypedDict):
    coords: Any
    attrs: Any
    dims: Any
    data_vars: Any
    encoding: Optional[Any]


class QuantumJob(pydantic.BaseModel):
    """Schema of the job data sent from the client"""

    tuid: str
    meas_return: MeasRet
    meas_level: MeasLvl
    meas_return_cols: int
    job_id: Optional[str] = None
    memory_slot_size: int = 100
    local: bool = True
    qobj: Optional[PulseQobj] = None
    metadata: Optional[QobjMetadata] = None
    header: Optional[QobjHeader] = None
    experiment_results: Dict[str, XArrayDict] = {}

    @pydantic.validator("metadata")
    def set_qobj_metadata(cls, v: Optional[QobjMetadata], values: dict, **kwargs):
        """Validator to set the metadata based on the qobj"""
        if "qobj" in values:
            return QobjMetadata.from_qobj(values["qobj"])
        return v

    @classmethod
    def from_hdf5(cls, file: Path, **kwargs):
        """Extract the quantum job from the hdf5 file

        The kwargs override any values defined in the file

        Args:
            file: the path to the file
            kwargs: extra key-word args
        """
        props = {}
        with h5py.File(file, mode="r") as hdf5_file:
            props["tuid"] = hdf5_file.attrs["tuid"]
            props["meas_return"] = MeasRet(hdf5_file.attrs["meas_return"])
            props["meas_level"] = MeasLvl(hdf5_file.attrs["meas_level"])
            props["meas_return_cols"] = hdf5_file.attrs["meas_return_cols"]
            props["header"] = hdf5_file["header"]
            props["experiment_results"] = hdf5_file["experiments"]

            if "job_id" in hdf5_file.attrs.keys():
                props["job_id"] = hdf5_file.attrs["job_id"]
                props["local"] = False

        return cls(**{**props, **kwargs})

    def to_hdf5(self, file: Path):
        """Saves this job to an HDF5 file

        Args:
            file: the path to the file where the data is to be saved
        """
        with h5py.File(file, mode="w") as hdf5_file:
            hdf5_file.attrs["tuid"] = self.tuid
            hdf5_file.attrs["meas_return"] = self.meas_return.value
            hdf5_file.attrs["meas_level"] = self.meas_level.value
            hdf5_file.attrs["meas_return_cols"] = self.meas_return_cols

            if self.job_id is not None:
                hdf5_file.attrs["job_id"] = self.job_id

            _save_header_to_hdf5(hdf5_file, self.header)
            _save_qobj_to_hdf5(hdf5_file, self.qobj)
            _save_results_to_hdf5(
                hdf5_file,
                self.experiment_results,
                meas_return_cols=self.meas_return_cols,
            )


def _save_header_to_hdf5(file: h5py.File, header: QobjHeader):
    """Saves the given header to the HDF5 file

    Args:
        file: the HDF5 file to save to
        header: the QobjHeader to save
    """
    header_data = header.to_dict()

    # save backend metadata
    backend_metadata = _get_subset_dict(header_data, keys=("backend_name",))
    _copy_hdf5_metadata(file, path="header/qobj/backend", source=backend_metadata)

    # save sweep metadata
    sweep_data = header_data.get("sweep", {})
    sweep_metadata = _get_subset_dict(
        sweep_data,
        keys=("dataset_name", "serial_order", "batch_size"),
        defaults={"batch_size": 1},
    )
    _copy_hdf5_metadata(file, path="header/qobj/sweep", source=sweep_metadata)

    # save slots metadata
    for path in parse.find(sweep_data, "slots"):
        sweep_group = file["header/qobj/sweep"]
        slots_grp = sweep_group.create_group("/".join(path))

        # -1 is "slots", -2 is parameter name, -3 is "parameters"
        param = path[-2]
        param_group_path = f"header/qobj/sweep/parameters/{param}"
        param_source = _get_subset_dict(
            sweep_data["parameters"][param], keys=("long_name", "unit")
        )
        _copy_hdf5_metadata(file, path=param_group_path, source=param_source)

        slots_dict = _get_value_at_path(sweep_data, path)
        # store all specified sweep parameter data in respective HDF datasets
        for slot_idx, slot_data in slots_dict.items():
            data = np.asarray(slot_data)
            dataset_key = f"slot{JOB_HDF5_FILE_DELIMITER}{slot_idx}"
            slots_grp.create_dataset(dataset_key, data=data)


def _save_qobj_to_hdf5(file: h5py.File, qobj: PulseQobj):
    """Saves the qobj as metadata in the HDF5 file

    Args:
        file: the HDF5 file to save to
        qobj: the qobject whose metadata is being saved
    """
    # save the raw metadata
    _copy_hdf5_metadata(
        file,
        path="header/qobj_metadata",
        source={
            "shots": qobj.config.shots,
            "qobj_id": qobj.qobj_id,
            "num_experiments": len(qobj.experiments),
        },
    )

    # save the raw experiments
    experiment_data = json.dumps(qobj.to_dict(), cls=PulseQobj_encoder, indent="\t")
    _copy_hdf5_metadata(
        file,
        path="header/qobj_data",
        source={"experiment_data": experiment_data},
    )


def _save_results_to_hdf5(
    file: h5py.File, results: Dict[str, XArrayDict], meas_return_cols: int
):
    """Saves the experiment results to the HDF5 file

    Args:
        file: the HDF5 file to save to
        results: the experiment results to save
        meas_return_cols: the meas_return_cols from the program settings
    """
    for name, result in results.items():
        path = f"experiments/{name}"
        data_vars = result["data_vars"]

        for acq_index, acq in enumerate(data_vars):
            channel = f"slot{JOB_HDF5_FILE_DELIMITER}{acq}"
            channel_path = f"{path}/{channel}"
            data_path = f"{channel_path}/measurement"

            # Get maximum acquisition index in each acquisition channel
            max_acq_idx = result["dims"][f"acq_index_{acq_index}"]

            # For each acqusition channel, create a corresponding measurement matrix
            # in a memory slot whose index corresponds to the acqusition channel,
            # unless it already exists.
            if channel_path not in file:
                # The columns of the matrix are the # of shots and the rows are
                # the horizontally composed measurements on the acqusition channel
                # (row indices corresponding to acquisition indices)
                dset = file.require_dataset(
                    data_path,
                    shape=(max_acq_idx, meas_return_cols),
                    dtype=complex,
                )
            else:
                dset = file[data_path]

            # save the data as complex number array
            data = data_vars[acq]
            tmp = np.zeros(len(data["data"]), dtype=complex)

            for idx in range(len(data["data"])):
                tmp.real[idx] = np.real(data["data"][idx])
                tmp.imag[idx] = np.imag(data["data"][idx])

            dset[...] = tmp


def _copy_hdf5_metadata(file: h5py.File, path: str, source: dict):
    """Copy the whole dict to HDF5 metadata for to the given group_path

    Args:
        file: the HDF5 file
        path: the /-separated path to the group
        source: the dictionary to copy from
    """
    if len(source) == 0:
        # do nothing if dict is empty
        return

    if path not in file:
        group = file.create_group(path)
    else:
        group = file[path]

    for key, value in source.items():
        group.attrs[key] = value


def _get_value_at_path(data: dict, path: List[str]) -> Any:
    """Retrieves the value at the given path of the nested dict data

    e.g. ["foo", "bar", "py"] return data["foo"]["bar"]["py"]

    Args:
        data: the nested dictionary
        path: the path to the value needed

    Returns:
        the value at the given path
    """
    value = data
    for part in path:
        value = data[part]

    return value


def _get_subset_dict(
    data: Dict[str, Any],
    keys: Tuple[str, ...],
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Gets a subset of a dictionary having the given keys if they exist"""
    if defaults is None:
        defaults = {}

    all_data = {**defaults, **data}
    return {k: all_data[k] for k in keys if k in all_data}

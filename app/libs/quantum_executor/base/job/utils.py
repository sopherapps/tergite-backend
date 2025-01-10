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

import pickle

import h5py
import numpy as np
import retworkx as rx
import xarray as xr
from tqdm.auto import tqdm

# Quantify dataset specification:
# https://quantify-quantify-core.readthedocs-hosted.com/en/v0.6.2/technical_notes/dataset_design/Quantify%20dataset%20-%20specification.html

"""
    Measurement types are determined as follows:  m<MeasLvl><MeasRet>
        m21 : Discriminated and Averaged measurement (TODO: Incorporate discriminator selection)
        m20 : Discriminated and Appended measurement (TODO: Incorporate discriminator selection)
        m11 : Integrated and Averaged measurement
        m10 : Integrated and Appended measurement
        m01 : Raw and Averaged measurement
        m00 : Unsupported measurement
"""


# -------------------------- LOADING FUNCTIONS -------------------------- #


def load_graph(data, experiment_name) -> rx.PyDiGraph:
    blob = bytes(data.experiments[experiment_name]["experiment_graph"][:])
    return pickle.loads(blob)


def load_coords(data) -> dict:
    """Determines which parameters are being swept and extracts their setpoint values.
    Repeat the procedure for every slot in the sweep.
    Coords are to the xarray dataset as settables are to Quantify.
    """
    dataset_coords = dict()
    for path in find(data.header["qobj/sweep"], "slots"):
        # traverse data.header["qobj/sweep"]
        slots = data.header["qobj/sweep"]
        for k in path:
            slots = slots[k]

        # extract parameter values for each slot
        for slot, slot_data in slots.items():
            dataset_coords["/".join((path[1], slot))] = slot_data

    return dataset_coords


def load_data(data) -> dict:
    """Helper function which extracts the acquisition data and returns
    it in a dictionary. The keys are the acquisitions and the values is the data.
    """
    dataset_vars = dict()

    for path in tqdm(data.sorted_measurements, ascii=" #", desc=data.tuid):
        # traverse data.experiments
        msmt = data.experiments
        for k in path:
            msmt = msmt[k]

        # there may be multiple acquisitions per channel
        for acq_idx, acq in enumerate(msmt):
            slot = path[-2]  # -1 is "measurement", -2 is slot tag
            if (key := "/".join((slot, f"acq~{acq_idx}"))) not in dataset_vars:
                dataset_vars[key] = list()
            dataset_vars[key].append(acq)  # relies on sorting

    keys = list(dataset_vars.keys())
    for k in keys:
        dataset_vars[k] = np.asarray(dataset_vars[k])
    return dataset_vars


# -------------------------- PARSING FUNCTIONS -------------------------- #


def appended_integrated(data) -> xr.Dataset:
    """Measurement type: m10
    These types of measurements contain a row of complex datapoints
    for every acquisition in every slot for every experiment.

    The column index in the row corresponds to the repeated "shot" value.

    e.g. if there are 1000 shots in the experiment, then there are 1000 elements in the row.
    """
    assert data.meas_level.value == 1
    assert data.meas_return.value == 0

    dataset_coords = load_coords(data)
    dataset_vars = load_data(data)

    dataset_coords["repetitions"] = np.arange(0, data.meas_return_cols, 1)

    return format_dataset(
        data,
        data_vars=format_vars(
            data, dataset_vars, outer_dims=[("repetitions", data.meas_return_cols)]
        ),
        coords=format_coords(data, dataset_coords),
    )


def averaged_raw(data) -> xr.Dataset:
    """Measurement type: m01
    These types of measurements contain a row of complex datapoints
    for every acquisition in every slot for every experiment.

    The column index in the row corresponds to the value of the readout as a time signal
    at that moment in time, with respect to the sample rate of the readout instruments.

    e.g. if the sample time is 1ns, then the element in the third column will correspond
    to the value of the readout signal 3ns after acquisition is initiated.
    """
    assert data.meas_level.value == 0
    assert data.meas_return.value == 1

    dataset_coords = load_coords(data)
    dataset_vars = load_data(data)

    # FIXME, hardcoded. Store in storage file
    # FIXME, relative t0 of acquisition, change coordinate to absolute time in schedule. Store in storage file
    dataset_coords["time"] = np.arange(0, data.meas_return_cols / 1e9, 1e-9)

    return format_dataset(
        data,
        data_vars=format_vars(
            data, dataset_vars, outer_dims=[("time", data.meas_return_cols)]
        ),
        coords=format_coords(data, dataset_coords),
    )


def averaged_integrated(data) -> xr.Dataset:
    """Measurement type: m11
    These types of measurements contain only a single complex datapoint
    for every acquisition in every slot for every experiment.
    """
    assert data.meas_level.value == 1
    assert data.meas_return.value == 1

    dataset_coords = load_coords(data)
    dataset_vars = load_data(data)

    return format_dataset(
        data,
        data_vars=format_vars(data, dataset_vars),
        coords=format_coords(data, dataset_coords),
    )


# -------------------------- XARRAY FORMATTING FUNCTIONS -------------------------- #


def format_dataset(data, *, data_vars: dict, coords: dict) -> xr.Dataset:
    return xr.Dataset(
        # dependent variables in dataset
        data_vars=data_vars,
        # independent variables in dataset
        coords=coords,
        # dataset attribute metadata
        attrs=data.file.attrs,
    )


def format_vars(data, dataset_vars: dict, *, outer_dims=None) -> dict:
    if outer_dims is None:
        outer_dims = list()
    names, shapes = tuple(zip(*outer_dims)) if len(outer_dims) else (tuple(), tuple())
    ret_str = "Average measured" if data.meas_return.value == 1 else "Measured"
    if data.meas_level.value == 2:
        lvl_str = "(Integrated and Discriminated)"
    elif data.meas_level.value == 1:
        lvl_str = "(Integrated)"
    elif data.meas_level.value == 0:
        lvl_str = "(Time signal)"

    return {
        tag: (
            _main_dims(data, tag) + names,
            y.reshape(_main_dims_shape(data, tag) + shapes),
            {
                "unit": "V",
                "long_name": f"{ret_str} signal value {lvl_str}",
                "is_main_var": True,
                #     "uniformly_spaced": True,
                #     "grid": True,
                #     "is_dataset_ref": False,
                "has_repetitions": data.meas_return.value == 0,
                #     "json_serialize_exclude": list(),
                "coords": _main_dims(data, tag) + names,
            },
        )
        for tag, y in dataset_vars.items()
    }


def format_coords(data, dataset_coords: dict) -> dict:
    return {
        coord: (
            coord,
            coord_data,
            {
                "unit": data.header[
                    f"qobj/sweep/parameters/{coord.split('/')[0]}"
                ].attrs["unit"],
                "long_name": data.header[
                    f"qobj/sweep/parameters/{coord.split('/')[0]}"
                ].attrs["long_name"],
                "is_main_coord": True,
                "uniformly_spaced": _is_uniform(coord_data),
                #    "is_dataset_ref": False,
                #    "json_serialize_exclude": list()
            },
        )
        for coord, coord_data in dataset_coords.items()
    }


def find(haystack, needle):
    if not isinstance(haystack, h5py.Group) and not isinstance(haystack, dict):
        return

    if needle in haystack:
        yield (needle,)

    for key, val in haystack.items():
        for subpath in find(val, needle):
            yield (key, *subpath)


# -------------------------- MISC. HELPER FUNCTIONS -------------------------- #
def _is_uniform(arr: np.array) -> bool:
    arr_diff = np.diff(arr)
    return all(np.abs(d - arr_diff[0]) < 1e-5 for d in arr_diff)


def _main_dims(data, slot_tag: str) -> tuple:
    """Helper function which returns the main dimension names of the parametric sweep for the given
    slot_tag, in the sweep's serial order.
    """
    slot_tag = slot_tag.split("/")[
        0
    ]  # ignore acquisition tag, sweeps are @ the slot level
    # all sweeps have serial orders
    serial_order = data.header["qobj/sweep"].attrs["serial_order"]
    return tuple("/".join((param, slot_tag)) for param in serial_order)


def _main_dims_shape(data, slot_tag: str) -> tuple:
    """Helper function which returns the shape of the parametric sweep for the given slot_tag."""
    slot_tag = slot_tag.split("/")[
        0
    ]  # ignore acquisition tag, sweeps are @ the slot level
    serial_order = data.header["qobj/sweep"].attrs["serial_order"]
    params = data.header["qobj/sweep/parameters"]
    return tuple(len(params[p]["/".join(("slots", slot_tag))]) for p in serial_order)

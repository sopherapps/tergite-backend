# This code is part of Tergite
#
# (C) Axel Andersson (2022)
# (C) Martin Ahindura (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import enum
from dataclasses import dataclass
from enum import Enum
from typing import Union, Type

import numpy as np
from qiskit.qobj import PulseQobjConfig
from quantify_scheduler.enums import BinMode


class MeasLvl(int, Enum):
    DISCRIMINATED = 2
    INTEGRATED = 1
    RAW = 0


class MeasRet(int, Enum):
    AVERAGED = 1
    APPENDED = 0


class MeasProtocol(str, enum.Enum):
    SSB_INTEGRATION_COMPLEX = "SSBIntegrationComplex"
    TRACE = "trace"


@dataclass(frozen=True)
class NativeQobjConfig:
    """Settings for running native experiments"""

    acq_return_type: Union[Type[complex], Type[np.ndarray]]
    protocol: MeasProtocol
    bin_mode: BinMode
    meas_level: MeasLvl
    meas_return: MeasRet
    meas_return_cols: int
    shots: int


def to_native_qobj_config(config: PulseQobjConfig) -> "NativeQobjConfig":
    """Converts the pulse qobj config to native qobj config

    Args:
        config: the configuration object of the pulse qobj

    Returns:
        NativeQobjConfig instance
    """
    try:
        bin_mode = _get_bin_mode(config)
        protocol = _get_meas_protocol(config)
        meas_level = MeasLvl(config.meas_level)

        if (
            bin_mode is BinMode.AVERAGE
            and protocol is MeasProtocol.SSB_INTEGRATION_COMPLEX
        ):
            return NativeQobjConfig(
                acq_return_type=complex,
                protocol=protocol,
                bin_mode=bin_mode,
                meas_level=meas_level,
                meas_return=MeasRet.AVERAGED,
                meas_return_cols=1,
                shots=config.shots,
            )

        if bin_mode is BinMode.AVERAGE and protocol is MeasProtocol.TRACE:
            return NativeQobjConfig(
                acq_return_type=np.ndarray,
                protocol=protocol,
                bin_mode=bin_mode,
                meas_level=meas_level,
                meas_return=MeasRet.AVERAGED,
                meas_return_cols=16384,  # length of a trace
                shots=config.shots,
            )

        if (
            bin_mode is BinMode.APPEND
            and protocol is MeasProtocol.SSB_INTEGRATION_COMPLEX
        ):
            return NativeQobjConfig(
                acq_return_type=np.ndarray,
                protocol=MeasProtocol.SSB_INTEGRATION_COMPLEX,
                bin_mode=BinMode.APPEND,
                meas_level=meas_level,
                meas_return=MeasRet.APPENDED,
                meas_return_cols=config.shots,
                shots=config.shots,
            )

    except KeyError:
        raise RuntimeError(
            f"Combination {(config.meas_return, config.meas_return)} is not supported."
        )


def _get_bin_mode(qobj_conf: PulseQobjConfig) -> BinMode:
    """Gets the BinMode based on the meas_return of the qobj.config

    Args:
        qobj_conf: the qobject config whose bin mode is to be obtained

    Returns:
        the BinMode for the given qobj
    """
    meas_return = qobj_conf.meas_return
    if isinstance(meas_return, int):
        return {
            int(MeasRet.APPENDED): BinMode.APPEND,
            int(MeasRet.AVERAGED): BinMode.AVERAGE,
        }[meas_return]

    # FIXME: For some reason, PulseQobjConfig expects to be an int
    #   yet our fixtures all have strings.
    meas_return = str.lower(qobj_conf.meas_return)
    return {
        "avg": BinMode.AVERAGE,
        "average": BinMode.AVERAGE,
        "averaged": BinMode.AVERAGE,
        "single": BinMode.APPEND,
        "append": BinMode.APPEND,
        "appended": BinMode.APPEND,
    }[meas_return]


def _get_meas_protocol(qobj_conf: PulseQobjConfig) -> "MeasProtocol":
    """Gets the measurement protocol for the given qobject

    Args:
        qobj_conf: the qobject config from which to extract the measurement protocol

    Returns:
        the measurement protocol for the given qobject
    """
    return {
        0: MeasProtocol.SSB_INTEGRATION_COMPLEX,
        1: MeasProtocol.TRACE,
        2: MeasProtocol.TRACE,
    }[qobj_conf.meas_level]

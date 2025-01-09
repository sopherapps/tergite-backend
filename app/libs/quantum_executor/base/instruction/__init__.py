# This code is part of Tergite
#
# (C) Axel Andersson (2022)
# (C) Martin Ahindura (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from typing import Dict, List

import numpy as np
from qiskit.qobj import PulseQobjConfig, PulseQobjInstruction
from quantify_scheduler.enums import BinMode

from .types import (
    MeasSettings,
    MeasProtocol,
    Instruction,
    DelayInstruction,
    ParamPulseInstruction,
    PhaseInstruction,
    FreqInstruction,
    PulseLibInstruction,
    AcquireInstruction,
)
from app.libs.quantum_executor.utils.general import ceil4
from app.libs.storage_file import MeasLvl, MeasRet


def get_meas_settings(config: PulseQobjConfig) -> "MeasSettings":
    """Gets the measurement settings given a pulse qobj config

    Args:
        config: the configuration object of the pulse qobj

    Returns:
        MeasSettings instance
    """
    try:
        bin_mode = _get_bin_mode(config)
        protocol = _get_meas_protocol(config)
        meas_level = MeasLvl(config.meas_level)

        if (
            bin_mode is BinMode.AVERAGE
            and protocol is MeasProtocol.SSB_INTEGRATION_COMPLEX
        ):
            return MeasSettings(
                acq_return_type=complex,
                protocol=protocol,
                bin_mode=bin_mode,
                meas_level=meas_level,
                meas_return=MeasRet.AVERAGED,
                meas_return_cols=1,
            )

        if bin_mode is BinMode.AVERAGE and protocol is MeasProtocol.TRACE:
            return MeasSettings(
                acq_return_type=np.ndarray,
                protocol=protocol,
                bin_mode=bin_mode,
                meas_level=meas_level,
                meas_return=MeasRet.AVERAGED,
                meas_return_cols=16384,  # length of a trace
            )

        if (
            bin_mode is BinMode.APPEND
            and protocol is MeasProtocol.SSB_INTEGRATION_COMPLEX
        ):
            return MeasSettings(
                acq_return_type=np.ndarray,
                protocol=MeasProtocol.SSB_INTEGRATION_COMPLEX,
                bin_mode=BinMode.APPEND,
                meas_level=meas_level,
                meas_return=MeasRet.APPENDED,
                meas_return_cols=config.shots,
            )

    except KeyError:
        raise RuntimeError(
            f"Combination {(config.meas_return, config.meas_return)} is not supported."
        )


def extract_instructions(
    qobj_inst: PulseQobjInstruction,
    config: PulseQobjConfig,
    hardware_map: Dict[str, str] = None,
) -> List[Instruction]:
    """Extracts tergite-specific instructions from the PulseQobjInstruction

    Args:
        qobj_inst: the PulseQobjInstruction from which instructions are to be extracted
        config: config of the pulse qobject
        hardware_map: the map describing the layout of the quantum device

    Returns:
        list of tergite-specific instructions
    """
    if hardware_map is None:
        hardware_map = {}

    name = qobj_inst.name
    t0 = ceil4(qobj_inst.t0) * 1e-9
    channel = qobj_inst.ch

    if name == "delay":
        return [
            DelayInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=ceil4(qobj_inst.duration) * 1e-9,
            )
        ]

    if name == "parametric_pulse":
        return [
            ParamPulseInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=ceil4(qobj_inst.parameters["duration"]) * 1e-9,
                pulse_shape=qobj_inst.pulse_shape,
                parameters=qobj_inst.parameters,
            )
        ]

    if name in ("setp", "fc"):  # "shiftf" is not working
        return [
            PhaseInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=0.0,
                phase=qobj_inst.phase,
            )
        ]

    if qobj_inst.name in ("setf",):  # "shiftf" is not working
        return [
            FreqInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                duration=0.0,
                frequency=qobj_inst.frequency * 1e9,
            )
        ]

    if qobj_inst.name in config.pulse_library:
        return [
            PulseLibInstruction(
                name=name,
                t0=t0,
                channel=channel,
                port=hardware_map.get(channel, channel),
                # FIXME: pulse_library seems to be a list but is accessed here as a dict
                duration=ceil4(config.pulse_library[name].shape[0]) * 1e-9,
            )
        ]

    if name == "acquire":
        program_settings = get_meas_settings(config)
        return [
            AcquireInstruction(
                name=name,
                t0=t0,
                channel=f"m{qubit_idx}",
                port=hardware_map.get(f"m{qubit_idx}", name),
                duration=ceil4(qobj_inst.duration) * 1e-9,
                memory_slot=qobj_inst.memory_slot[n],
                protocol=program_settings.protocol.value,
                acq_return_type=program_settings.acq_return_type,
                bin_mode=program_settings.bin_mode,
            )
            for n, qubit_idx in enumerate(qobj_inst.qubits)
        ]

    raise RuntimeError(f"No mapping for PulseQobjInstruction {qobj_inst}")


def _get_bin_mode(qobj_conf: PulseQobjConfig) -> BinMode:
    """Gets the BinMode based on the meas_return of the qobj.config

    Args:
        qobj_conf: the qobject config whose bin mode is to be obtained

    Returns:
        the BinMode for the given qobj
    """
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

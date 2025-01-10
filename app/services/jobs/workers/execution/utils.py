# This code is part of Tergite
#
# (C) Copyright Martin Ahindura 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simple utility functions for the execution worker"""
import settings
from app.libs.properties import initialize_backend, get_backend_config
from app.libs.quantum_executor.base.executor import QuantumExecutor
from app.libs.quantum_executor.qiskit.executor import (
    QiskitPulse1QExecutor,
    QiskitPulse2QExecutor,
    QiskitDynamicsExecutor,
)
from app.libs.quantum_executor.quantify.executor import QuantifyExecutor
from app.utils.http import get_mss_client


def get_executor(
    executor_type: str = settings.EXECUTOR_TYPE,
    config_file: str = settings.QUANTIFY_CONFIG_FILE,
    mss_url: str = settings.MSS_MACHINE_ROOT_URL,
) -> QuantumExecutor:
    """Gets the executor for running jobs

    It also initializes the backend before returning the executor

    Args:
        executor_type: the executor type to return
        config_file: the path to the configuration file of the executor
        mss_url: the URL to MSS

    Returns:
        An initialized quantum executor
    """
    executor = None  # type: QuantumExecutor
    backend_config = get_backend_config()
    qubit_config = None
    resonator_config = None
    discriminator_config = None
    coupler_config = None

    if executor_type == "quantify":
        executor = QuantifyExecutor(config_file=config_file)

    if executor_type == "qiskit_pulse_1q":
        executor: QiskitPulse1QExecutor = QiskitPulse1QExecutor(
            backend_config=backend_config
        )
        discriminator_config = executor.backend.train_discriminator()

    if executor_type == "qiskit_pulse_2q":
        executor: QiskitPulse2QExecutor = QiskitPulse2QExecutor(
            backend_config=backend_config
        )
        discriminator_config = executor.backend.train_discriminator()

    initialize_backend(
        mss_client=get_mss_client(),
        mss_url=mss_url,
        backend_config=backend_config,
        qubit_config=qubit_config,
        resonator_config=resonator_config,
        discriminator_config=discriminator_config,
        coupler_config=coupler_config,
    )

    return executor

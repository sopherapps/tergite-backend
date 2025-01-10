# This code is part of Tergite
#
# (C) Pontus Vikstål, Stefan Hill (2024)
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
from typing import Optional, Dict, Any, List

from qiskit.providers.models import PulseBackendConfiguration, PulseDefaults
from qiskit.transpiler import Target
from qiskit_dynamics import DynamicsBackend, Solver

from app.libs.properties import BackendConfig


class QiskitPulseBackend(DynamicsBackend):
    """Backend for running simulators on using QiskitDynamics"""

    def __init__(self, backend_config: BackendConfig, **options):
        self.backend_config = backend_config
        self.backend_name = backend_config.general_config.name

        options["backend_config"] = backend_config

        solver = self.generate_solver(**options)
        target = self.generate_target(**options)
        solver_options = self.generate_solver_options(**options)
        configuration = self.generate_configuration(**options)
        defaults = self.generate_pulse_defaults(**options)
        subsystem_dims = self.generate_subsystem_dims(**options)

        super().__init__(
            solver=solver,
            target=target,
            solver_options=solver_options,
            configuration=configuration,
            defaults=defaults,
            subsystem_dims=subsystem_dims,
            **options,
        )

    @classmethod
    @abc.abstractmethod
    def generate_solver(cls, **kwargs) -> Solver:
        """Generates the solver to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_configuration(cls, **kwargs) -> Optional[PulseBackendConfiguration]:
        """Generates the PulseBackendConfiguration to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_target(cls, **kwargs) -> Optional[Target]:
        """Generates the Target to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_solver_options(cls, **kwargs) -> Dict[str, Any]:
        """Generates the solver options to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_pulse_defaults(cls, **kwargs) -> Optional[PulseDefaults]:
        """Generates the PulseDefault's to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_subsystem_dims(cls, **kwargs) -> Optional[List[int]]:
        """Generates the subsystem_dims to pass to the backend when initializing"""
        pass

    @abc.abstractmethod
    def train_discriminator(self, shots: int = 1024, **kwargs):
        """
        Generates |0> and |1> states, trains a linear discriminator
        Args:
            shots: number of shots for generating i q data
            kwargs: extra key-word args

        Returns:
            Discriminator object as json in the format to store it in the database
        """
        pass

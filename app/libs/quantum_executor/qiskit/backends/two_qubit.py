# This code is part of Tergite
#
# (C) Pontus Vikstål, Stefan Hill (2024)
# (C) Chalmers Next Labs (2024)
# (C) Martin Ahindura (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from typing import List, Dict, Any, Optional

import numpy as np
from qiskit.providers.models import PulseDefaults, PulseBackendConfiguration, GateConfig
from qiskit.pulse import Schedule, Acquire, AcquireChannel, MemorySlot
from qiskit.quantum_info import Statevector
from qiskit.transpiler import Target
from qiskit_dynamics import Solver
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from app.libs.properties import BackendConfig
from app.libs.quantum_executor.qiskit.backends.base import QiskitPulseBackend
from app.libs.quantum_executor.qiskit.functions import omega_c


class QiskitPulse2Q(QiskitPulseBackend):
    r"""Backend for pulse simulations on a two-qubit transmon system.

    Args:
        backend_config: The configuration of this backend.
        r (float): The Rabi rate in Hz. Default is 1e9.
        atol (float): The absolute tolerance for the numerical solver. Default is 1e-6.
        rtol (float): The relative tolerance for the numerical solver. Default is 1e-6.
        dim (int): The dimension of the Hilbert space for each qubit. Default is 4.
        noise (bool): Flag to enable or disable noise in the simulation. Default is True.
        **options: Additional keyword arguments passed to the DynamicsBackend.
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        r: float = 1e9,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        dim: int = 3,
        noise: bool = True,
        **options,
    ):
        super().__init__(
            backend_config=backend_config,
            r=r,
            atol=atol,
            rtol=rtol,
            dim=dim,
            noise=noise,
            **options,
        )

    @classmethod
    def generate_subsystem_dims(cls, dim: int = 4, **kwargs) -> List[int]:
        return [dim, dim, dim]

    @classmethod
    def generate_solver_options(
        cls, backend_config: BackendConfig, **kwargs
    ) -> Dict[str, Any]:
        return {
            "method": "scipy_expm",
            "max_dt": backend_config.general_config.dt,
        }

    @classmethod
    def generate_pulse_defaults(
        cls, backend_config: BackendConfig, **kwargs
    ) -> Optional[PulseDefaults]:
        return None

    @classmethod
    def generate_target(cls, **kwargs) -> Optional[Target]:
        return None

    @classmethod
    def generate_configuration(
        cls, backend_config: BackendConfig, dim: int = 4, **kwargs
    ) -> PulseBackendConfiguration:
        backend_name = backend_config.general_config.name
        backend_version = backend_config.general_config.version
        dt = backend_config.general_config.dt

        num_qubits = backend_config.general_config.num_qubits
        if num_qubits < 2:
            raise ValueError("The backend configuration must have at least two qubits.")

        qubit_confs = backend_config.simulator_config.qubit
        coupler_confs = backend_config.simulator_config.coupler

        # For qubit 0
        qubit0_conf = qubit_confs[0]
        f0 = qubit0_conf["frequency"]

        # For qubit 1
        qubit1_conf = qubit_confs[1]
        f1 = qubit1_conf["frequency"]
        alpha1 = qubit1_conf.get("anharmonicity", -0.17e9)

        # TODO: propagate arguments from backend_config.toml

        coupler0_conf = coupler_confs[0]

        # Coupler frequency
        f2 = coupler0_conf["frequency"]

        # Coupling anharmonicity (in Hz)
        alpha2 = coupler0_conf["anharmonicity"]

        # Coupling strengths (in Hz)
        g02 = coupler0_conf["coupling_strength_02"]

        gates_configs = [
            GateConfig(name=k, **v) for k, v in backend_config.gates.items()
        ]

        return PulseBackendConfiguration(
            backend_name=backend_name,
            backend_version=backend_version,
            n_qubits=num_qubits,
            basis_gates=list(backend_config.gates.keys()),
            gates=gates_configs,
            local=True,
            simulator=backend_config.general_config.simulator,
            conditional=False,
            open_pulse=backend_config.general_config.open_pulse,
            memory=True,
            max_shots=4000,
            coupling_map=backend_config.device_config.coupling_map,
            meas_map=backend_config.device_config.meas_map,
            n_uchannels=1,
            u_channel_lo=[[f2 / 1e9 - 0.1, f2 / 1e9 + 0.1]],
            meas_levels=[1, 2],
            qubit_lo_range=[
                [f0 / 1e9 - 0.1, f0 / 1e9 + 0.1],
                [f1 / 1e9 - 0.1, f1 / 1e9 + 0.1],
                [f2 / 1e9 - 0.1, f2 / 1e9 + 0.1],
            ],  # in GHz
            meas_lo_range=[],
            dt=dt / 1e-9,  # in nanoseconds
            dtm=dt / 1e-9,  # in nanoseconds
            rep_times=[],
            meas_kernels=["boxcar"],
            discriminators=["lda"],
            description=backend_config.general_config.description,
            hamiltonian={
                "h_str": [
                    "2*np.pi*f1*N0",
                    "np.pi*alpha1*N0*N0",
                    "-np.pi*alpha1*N0",
                    "2*np.pi*f2*N1",
                    "np.pi*alpha2*N1*N1",
                    "-np.pi*alpha2*N1",
                    "2*np.pi*J*(a0+a0.dag())*(a1+a1.dag())",
                    "X0||D0",
                    "X1||D1",
                ],
                "vars": {
                    "f1": f1,
                    "alpha1": alpha1,
                    "f2": f2,
                    "alpha2": alpha2,
                    "J": g02,
                },
                "qub": {"0": dim, "1": dim},
                "osc": {},
                "description": "A two-qubit transmon Hamiltonian with 4 levels per qubit",
            },
        )

    @classmethod
    def generate_solver(
        cls,
        backend_config: BackendConfig,
        r: float = 1e9,
        dim: int = 4,
        **kwargs,
    ) -> Solver:
        dt = backend_config.general_config.dt

        qubit_confs = backend_config.simulator_config.qubit
        coupler_confs = backend_config.simulator_config.coupler

        # For qubit 0
        qubit0_conf = qubit_confs[0]
        f0 = qubit0_conf["frequency"]
        alpha0 = qubit0_conf.get("anharmonicity", -0.17e9)

        # For qubit 1
        qubit1_conf = qubit_confs[1]
        f1 = qubit1_conf["frequency"]
        alpha1 = qubit1_conf.get("anharmonicity", -0.17e9)

        # Extract coupling strength J from backend_config

        a = np.diag(np.sqrt(np.arange(1, dim)), 1)  # annihilation operator
        adag = np.diag(np.sqrt(np.arange(1, dim)), -1)  # creation operator
        N = np.diag(np.arange(dim))  # number operator

        # Define tensor products for two-qubit system
        I = np.eye(dim)

        # Order of modes: coupler (mode 3), qubit 2 (mode 2), qubit 1 (mode 1)
        a0 = np.kron(I, np.kron(I, a))
        a1 = np.kron(I, np.kron(a, I))
        a2 = np.kron(a, np.kron(I, I))

        adag0 = np.kron(I, np.kron(I, adag))
        adag1 = np.kron(I, np.kron(adag, I))
        adag2 = np.kron(adag, np.kron(I, I))

        N0 = np.kron(I, np.kron(I, N))
        N1 = np.kron(I, np.kron(N, I))
        N2 = np.kron(N, np.kron(I, I))

        ident = np.eye(dim**3, dtype=complex)

        # TODO: propagate arguments from backend_config.toml

        coupler0_conf = coupler_confs[0]

        # Coupler frequency
        f2 = coupler0_conf["frequency"]

        # Coupling anharmonicity (in Hz)
        alpha2 = coupler0_conf["anharmonicity"]

        # Coupling strengths (in Hz)
        g02 = coupler0_conf["coupling_strength_02"]
        g12 = coupler0_conf["coupling_strength_12"]

        # Create static Hamiltonian
        coupler_f_vs_t = omega_c(
            0,
            delta_0=coupler0_conf["cz_pulse_amplitude"],  # Maximum delta value
            theta=coupler0_conf["cz_pulse_dc_bias"],  # DC bias term
            omega_c0=2 * np.pi * f2,  # Maximum frequency in Hz
            omega_phi=coupler0_conf["frequency_detuning"]
            + 2 * np.pi * (f1 - f0 - alpha0),  # Transition frequency in Hz
            phi=coupler0_conf["cz_pulse_phase_offset"],  # Phase offset
            t_w=coupler0_conf["cz_pulse_duration_before"],  # s, duration before pulse
            t_rf=coupler0_conf["cz_pulse_duration_rise"],  # s, rise time
            t_p=coupler0_conf[
                "cz_pulse_duration_constant"
            ],  # s, constant pulse duration
        )

        static_ham = (
            2 * np.pi * f0 * N0
            + np.pi * alpha0 * N0 @ (N0 - ident)
            + 2 * np.pi * f1 * N1
            + np.pi * alpha1 * N1 @ (N1 - ident)
            + coupler_f_vs_t * N2
            + np.pi * alpha2 * N2 @ (N2 - ident)
            + 2 * np.pi * g02 * ((a0 + adag0) @ (a2 + adag2))
            + 2 * np.pi * g12 * ((a1 + adag1) @ (a2 + adag2))
        )

        # Create drive operators
        drive_op1 = 2 * np.pi * r * (a0 + adag0)
        drive_op2 = 2 * np.pi * r * (a1 + adag1)
        drive_op3 = N2

        return Solver(
            static_hamiltonian=static_ham,
            hamiltonian_operators=[drive_op1, drive_op2, drive_op3],
            hamiltonian_channels=["d0", "d1", "u0"],
            channel_carrier_freqs={"d0": f0, "d1": f1, "u0": 0},
            dt=dt,
            array_library="numpy",
            rotating_frame=static_ham,
            in_frame_basis=False,
        )

    def train_discriminator(self, shots: int = 1024, dim: int = 3, **kwargs):
        """
        Generates |0> and |1> states for each qubit, trains a linear discriminator

        FIXME: Find out and document what dim is for

        Args:
            shots: number of shots for generating data
            dim: ....

        Returns:
            Discriminator object as json in the format to store it in the database
        """
        # Generate the iq values for each qubit
        qubit_ids = [x[0] for x in self.configuration().meas_map]

        lda_results = {}
        for qubit_id in qubit_ids:
            schedule = Schedule(
                (0, Acquire(1, AcquireChannel(qubit_id), MemorySlot(0)))
            )

            # Process the solution to extract populations of interest
            # Map basis states to indices in the state vector
            def basis_state_index(n_tuple, dim):
                n0, n1, n2 = n_tuple
                return n2 * dim**2 + n1 * dim + n0

            def get_index(qubit_id, mode, dim):
                state_ind = [0] * dim
                state_ind[qubit_id] = mode
                return basis_state_index(tuple(state_ind), dim)

            def get_statevector(mode):
                sv = np.zeros(dim**3, dtype=complex)
                sv[get_index(qubit_id, mode, dim)] = 1
                return Statevector(sv)

            # For |0> state
            y0 = get_statevector(0)
            job_0 = self.run([schedule], shots=shots, initial_state=y0)
            i_q_values_0 = job_0.result().data()["memory"].reshape(shots, 2)

            # For |1> state
            y0 = get_statevector(1)
            job_1 = self.run([schedule], shots=shots, initial_state=y0)
            i_q_values_1 = job_1.result().data()["memory"].reshape(shots, 2)

            # Combine IQ values and labels
            combined_i_q_values = np.vstack((i_q_values_0, i_q_values_1))
            labels = np.concatenate((np.zeros(shots), np.ones(shots)))

            lda_model = LinearDiscriminantAnalysis()
            lda_model.fit(combined_i_q_values, labels)

            lda_intercepts = lda_model.intercept_.tolist()
            lda_coefs = lda_model.coef_.tolist()

            lda_result = {}
            for i in range(len(lda_intercepts)):
                lda_result["intercept"] = lda_intercepts[i]
                lda_result["coef_0"] = lda_coefs[i][0]
                lda_result["coef_1"] = lda_coefs[i][1]

            lda_results[qubit_id] = lda_result
        # Return the discriminator configuration
        return {"lda": lda_results}

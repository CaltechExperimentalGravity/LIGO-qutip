import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.constants import c  # Speed of light from SciPy constants
import timeit
import argparse
import yaml

# Parameter dictionary at the top of the file
parameters = {
    "lambda_laser": 1064e-9,  # Wavelength of the laser in meters
    "omega_0": 2 * np.pi * 1e5,  # Reduced angular frequency for computational stability
    "L": 4000,  # Arm cavity length in meters
    "T_ITM": 0.014,  # ITM Transmission (1.4%)
    "T_SRM": 0.323,  # SRM Transmission (32.3%)
    "R_SRM": 1 - 0.323,  # SRM Reflectivity (67.7%)
    "gamma": 100,  # Reduced decay rate (Hz) for photon accumulation
    "epsilon_drive": 5.0,  # Driving strength
    "N_fock": 10,  # Number of Fock states for each cavity
    "kappa": 0.005,  # Coupling rate relative to omega_0
    "tlist_end": 50e-3,  # End of time evolution (50 ms)
    "tlist_points": 200,  # Number of time points
    "output_file": "DARM_SRC_dynamics.png",  # Output filename for the plot
    "solver_configs": [
        {"nsteps": 50000, "atol": 1e-8, "rtol": 1e-6, "method": "adams"},
        {"nsteps": 100000, "atol": 1e-8, "rtol": 1e-6, "method": "bdf"}
    ],
    "optimal_solver_file": "optimal_solver.yaml"  # File to save optimal solver configuration
}

def main(args):
    # Modify parameters based on command line arguments
    if args.gamma is not None:
        parameters["gamma"] = args.gamma
    if args.epsilon_drive is not None:
        parameters["epsilon_drive"] = args.epsilon_drive
    if args.N_fock is not None:
        parameters["N_fock"] = args.N_fock
    if args.tlist_end is not None:
        parameters["tlist_end"] = args.tlist_end
    if args.tlist_points is not None:
        parameters["tlist_points"] = args.tlist_points

    # Quantum operators
    N = parameters["N_fock"]
    a = qt.destroy(N)  # Annihilation operator for the main interferometer cavity
    a_dag = a.dag()  # Creation operator
    n_a = a_dag * a  # Number operator

    b = qt.destroy(N)  # Annihilation operator for SRM cavity
    b_dag = b.dag()  # Creation operator for SRM cavity
    n_b = b_dag * b  # Number operator

    # Hamiltonians
    omega_0 = parameters["omega_0"]
    epsilon_drive = parameters["epsilon_drive"]

    # Convert drive strength to appropriate Hamiltonian units
    H_drive = (epsilon_drive) * (qt.tensor(a_dag, qt.qeye(N)) + qt.tensor(a, qt.qeye(N)))

    # Main interferometer cavity Hamiltonian
    H_interferometer = omega_0 * qt.tensor(n_a, qt.qeye(N))
    H_srm = omega_0 * qt.tensor(qt.qeye(N), n_b)

    # Interaction between interferometer and SRM
    kappa = parameters["kappa"]
    H_int = kappa * omega_0 * (qt.tensor(a, b_dag) + qt.tensor(a_dag, b))

    # Total Hamiltonian for the two-mode system (interferometer + SRM)
    H = H_interferometer + H_srm + H_int + H_drive

    # Collapse operators
    gamma = parameters["gamma"]  # Photon decay rate in Hz
    c_ops = [np.sqrt(gamma) * qt.tensor(a, qt.qeye(N))]

    # Initial state (vacuum state for both cavities)
    psi0 = qt.tensor(qt.basis(N, 0), qt.basis(N, 0))

    # Time evolution list
    tlist = np.linspace(0, parameters["tlist_end"], parameters["tlist_points"])

    # Solver options to try out
    solver_configs = parameters["solver_configs"]
    optimal_solver = solver_configs[0]  # Use the first solver configuration by default

    # Solve the system dynamics with the chosen solver
    if args.verbose:
        # If verbose flag is set, measure solver execution time
        start_time = timeit.default_timer()
        try:
            result = qt.mesolve(H, psi0, tlist, c_ops, [qt.tensor(n_a, qt.qeye(N)), qt.tensor(qt.qeye(N), n_b)], options=optimal_solver)
            elapsed_time = timeit.default_timer() - start_time
            print(f"Solver execution time: {elapsed_time:.4f} seconds")
        except Exception as e:
            print(f"Solver attempt failed: {e}")
            return
    else:
        # Run solver without timing
        try:
            result = qt.mesolve(H, psi0, tlist, c_ops, [qt.tensor(n_a, qt.qeye(N)), qt.tensor(qt.qeye(N), n_b)], options=optimal_solver)
        except Exception as e:
            print(f"Solver attempt failed: {e}")
            return

    # Plotting and saving the results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tlist, result.expect[0], label="Interferometer Cavity (⟨n_a⟩)")
    ax.plot(tlist, result.expect[1], label="SRM Cavity (⟨n_b⟩)")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Photon Number Expectation Value')
    ax.set_title('Dynamics of LIGO Interferometer with SRM')
    ax.legend()
    ax.grid()

    # Save plot to a PNG file with the specified name
    fig.savefig(parameters["output_file"])
    print(f"Plot saved as {parameters['output_file']}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate LIGO interferometer dynamics with SRM.")
    parser.add_argument('--gamma', type=float, help="Set the decay rate (gamma in Hz)")
    parser.add_argument('--epsilon_drive', type=float, help="Set the driving strength (epsilon_drive in photons per second)")
    parser.add_argument('--N_fock', type=int, help="Set the number of Fock states (N_fock)")
    parser.add_argument('--tlist_end', type=float, help="Set the end time for the time evolution (seconds)")
    parser.add_argument('--tlist_points', type=int, help="Set the number of time points for the evolution")
    parser.add_argument('--verbose', action='store_true', help="Print solver execution time")

    args = parser.parse_args()
    main(args)

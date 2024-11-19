# Rodeo Algorithm implementation in TenPY. Does not calculate the exact resulting state but
# returns success probability and overlap with a given target state.

import numpy as np
from tenpy.algorithms import dmrg, tebd

# Get the intermediate times for measurements to get overlaps.
def get_measurement_times(time_samples):
    measurement_times = []
    for i in range(2**len(time_samples)):
        measurement_time = 0
        binary_string_coeffs = bin(i)[2:].rjust(len(time_samples), '0')
        for i in range(len(binary_string_coeffs)):
            measurement_time += int(binary_string_coeffs[i]) * time_samples[i]
        measurement_times.append(measurement_time)
    return sorted(measurement_times)

# Do a rodeo run with the given time samples
# Does not calculate the exact resulting state, and only calculates overlap
# O(2^(rodeo_cycles)) so slows down a lot for rodeo_cycles > 6
# Returns the overlap squared with the target state and the success probability
def rodeo_run_old(hamiltonian, input_state, target_state, E_target, time_samples, tebd_params):
    if len(time_samples) == 0:
        # In this case, there are 0 circuits of the RA
        zero_input_state = input_state.copy()
        zero_norm_sq = zero_input_state.overlap(zero_input_state)

        zero_overlap = target_state.copy().overlap(zero_input_state)

        return zero_overlap.conj() * zero_overlap/zero_norm_sq, zero_norm_sq

    time_evolution_engine = tebd.TEBDEngine(input_state.copy(), hamiltonian, tebd_params) 

    measurement_times = get_measurement_times(time_samples)
    measurement_overlaps = []
    coeffs_for_f_operators = []
    f_operators_without_coeffs = []
    while time_evolution_engine.evolved_time < measurement_times[-1]:
        for measurement_time in measurement_times:
            if time_evolution_engine.evolved_time <= measurement_time and measurement_time <= tebd_params['dt'] + time_evolution_engine.evolved_time:
                measurement_overlap = target_state.overlap(time_evolution_engine.psi)
                measurement_overlap *= np.exp(1j * E_target * measurement_time)
                coeffs_for_f_operators.append(np.exp(1j * E_target * measurement_time))
                f_operators_without_coeffs.append(time_evolution_engine.psi.copy())
                measurement_overlaps.append(measurement_overlap)
        time_evolution_engine.run()
   
    norm_sq = 0
    for i in range(len(coeffs_for_f_operators)):
        for j in range(len(coeffs_for_f_operators)):
            norm_sq += np.conj(coeffs_for_f_operators[i]) * coeffs_for_f_operators[j] * f_operators_without_coeffs[i].overlap(f_operators_without_coeffs[j])
    norm_sq /= (len(measurement_times)**2)
    overlap = np.sum(measurement_overlaps)/len(measurement_overlaps)
    return  overlap.conj() * overlap/norm_sq , norm_sq

# Do a rodeo run with the given time samples
# Returns the overlap squared with the target state and the success probability
def rodeo_run(hamiltonian, input_state, target_state, E_target, time_samples, tebd_params):
    if len(time_samples) == 0:
        # In this case, there are 0 circuits of the RA
        zero_input_state = input_state.copy()
        zero_norm_sq = zero_input_state.overlap(zero_input_state)

        zero_overlap = target_state.copy().overlap(zero_input_state)

        return zero_overlap.conj() * zero_overlap/zero_norm_sq, zero_norm_sq

    current_psi = input_state.copy()
    for time_sample in time_samples:
        current_psi = single_rodeo_step(hamiltonian, current_psi, E_target, time_sample, tebd_params)

    norm_sq = current_psi.overlap(current_psi)
    overlap = current_psi.overlap(target_state)
    return overlap.conj() * overlap / norm_sq, norm_sq



# returns the MPS after one cycle of the Rodeo Algorithm
def single_rodeo_step(hamiltonian, input_state, E_target, time_step, tebd_params):
    # Round the time step to nearest trotter step value
    alpha = 1/2
    beta = 1/2 * np.exp(1j * E_target * time_step)

    time_evolution_engine_one_step = tebd.TEBDEngine(input_state.copy(), hamiltonian, tebd_params)

    while time_evolution_engine_one_step.evolved_time < time_step-0.0001:
        time_evolution_engine_one_step.run()
        print(f"{time_evolution_engine_one_step.evolved_time/time_step}")

    return input_state.copy().add(time_evolution_engine_one_step.psi, alpha, beta)

    """
    measurement_times = get_measurement_times(time_samples)
    measurement_overlaps = []
    coeffs_for_f_operators = []
    f_operators_without_coeffs = []
    while time_evolution_engine.evolved_time < measurement_times[-1]:
        for measurement_time in measurement_times:
            if time_evolution_engine.evolved_time <= measurement_time and measurement_time <= tebd_params['dt'] + time_evolution_engine.evolved_time:
                measurement_overlap = target_state.overlap(time_evolution_engine.psi)
                measurement_overlap *= np.exp(1j * E_target * measurement_time)
                coeffs_for_f_operators.append(np.exp(1j * E_target * measurement_time))
                f_operators_without_coeffs.append(time_evolution_engine.psi.copy())
                measurement_overlaps.append(measurement_overlap)
        time_evolution_engine.run()
   
    norm_sq = 0
    for i in range(len(coeffs_for_f_operators)):
        for j in range(len(coeffs_for_f_operators)):
            norm_sq += np.conj(coeffs_for_f_operators[i]) * coeffs_for_f_operators[j] * f_operators_without_coeffs[i].overlap(f_operators_without_coeffs[j])
    norm_sq /= (len(measurement_times)**2)
    overlap = np.sum(measurement_overlaps)/len(measurement_overlaps)
    return  overlap.conj() * overlap/norm_sq , norm_sq
    """


def rodeo_run_gaussian_sample(hamiltonian, input_state, target_state, E_target, sigma, r, tebd_params):
    time_samples = np.abs(sigma*np.random.randn(r)) # Absolute value as TEBD doesn't work with negative times
    return rodeo_run(hamiltonian, input_state, target_state, E_target, time_samples, tebd_params)

def rodeo_run_gaussian_sample_old(hamiltonian, input_state, target_state, E_target, sigma, r, tebd_params):
    time_samples = np.abs(sigma*np.random.randn(r)) # Absolute value as TEBD doesn't work with negative times
    return rodeo_run_old(hamiltonian, input_state, target_state, E_target, time_samples, tebd_params)

def main():
    # Return bound indicies that do not have an interaction initially, based on original chain.
    def get_non_interaction_term_indicies(initial_state):
        indicies = []

        i = -1
        for block_length in initial_state:
            i+=block_length
            indicies.append(i)

        return indicies[:-1]
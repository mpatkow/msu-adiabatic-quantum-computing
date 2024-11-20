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


    from tqdm import tqdm
    import tenpy
    from tenpy.models.xxz_chain import XXZChain2 as XXZChain
    import matplotlib.pyplot as plt

    # Model parameters
    J = 1
    SHAPE = [2]*2
    L = sum(SHAPE)
    mu = 0
    Jz = -0.3

    # Simulation parameters
    E_target_vals = np.linspace(-2,-0,15)
    sigma = 1
    r_vals = [5] 
    resamples = 1

    dmrg_params = {
        'mixer': None,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-10,
        },
        'combine': True,
    }

    tebd_params = {
        'N_steps': 1,
        'dt': 0.1,
        'order': 4,
        'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
    }

    # Construct initial hamiltonian with missing coupling terms
    c_arr = np.ones(L-1)
    for non_coupling_index in get_non_interaction_term_indicies(SHAPE):
        c_arr[non_coupling_index] = 0
    initial_ground_hamiltonian_dmrg = XXZChain({'Jxx':J*c_arr,"Jz":Jz,'hz':mu,"L":L})

    # Hamiltonian used for Rodeo Alg.
    rodeo_H = XXZChain({'Jxx':J,"Jz":Jz,'hz':mu,"L":L})

    # Guess for the ground state of the initial_model
    initial_state_guess = tenpy.networks.mps.MPS.from_lat_product_state(initial_ground_hamiltonian_dmrg.lat, [['up'], ['down']])

    # Use DMRG to calculate initial ground state of uncoupled model
    dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(initial_state_guess, initial_ground_hamiltonian_dmrg, dmrg_params)
    E0_uncoupled, initial_state = dmrg_eng_uncoupled_state.run()

    import new_model_tenpy as nmt
    AMI = nmt.AdiabaticHamiltonian({"Jxx": -4, "Jz":Jz, "hz":0, "L":L})
    AMF = XXZChain({'Jxx':J,"Jz":Jz,'hz':mu,"L":L})
    initial_state = nmt.complete_adiabatic_evolution_run(AMI, AMF, dmrg_params, tebd_params,1)

    # Take a copy of the initial state for later overlaps just in case it changes.
    #psi_i_for_i_overlap = initial_state.copy()

    # Get the exact ground state of the final Hamiltonian with DMRG
    # In reality we would not know this, but we use this to calculate overlaps
    dmrg_eng_final_state = dmrg.TwoSiteDMRGEngine(initial_state.copy(), rodeo_H, dmrg_params)
    E0_coupled, goal_state = dmrg_eng_final_state.run() 

    #print(goal_state.overlap(psi_i_for_i_overlap))
    #print(goal_state.overlap(goal_state))

    y_vals_FAKE = []
    for r in r_vals:
        y_vals = []
        y_vals_2 = []
        #time_samples = np.abs(sigma*np.random.randn(r)) # Absolute value as TEBD doesn't work with negative times
        for resample_i in tqdm(range(resamples)):
            y_val_set = []
            y_val_set_2 = []
            time_samples = np.abs(sigma*np.random.randn(r)) # Absolute value as TEBD doesn't work with negative times
            for E_target in tqdm(E_target_vals):
                #time_samples = np.abs(sigma*np.random.randn(r)) # Absolute value as TEBD doesn't work with negative times
                RENAME_DATA,RENAME_DATA_2 = rodeo_run(rodeo_H, initial_state, goal_state, E_target, time_samples, tebd_params)
                y_val_set.append(RENAME_DATA)
                y_val_set_2.append(RENAME_DATA_2)
            y_vals.append(y_val_set)
            y_vals_2.append(y_val_set_2)
        y_vals_FAKE.append(np.mean(y_vals,axis=0)[0])
        #print(y_vals_FAKE)

        plt.plot(E_target_vals,np.mean(y_vals,axis=0),label="overlap")
        plt.plot(E_target_vals,np.mean(y_vals_2,axis=0),label="success_chance")
        plt.legend()
        plt.show()
    plt.plot(r_vals, y_vals_FAKE)
    plt.show()

if __name__ == "__main__":
    main() 

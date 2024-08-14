# Rodeo Algorithm implementation in TenPY, WIP as of Aug. 6 2024

import numpy as np
import scipy
import tenpy
import matplotlib.pyplot as plt
from tenpy.algorithms import dmrg, tebd
from tenpy.models.xxz_chain import XXZChain2 as XXZChain

# Return bound indicies that do not have an interaction initially, based on original chain.
def get_non_interaction_term_indicies(initial_state):
    indicies = []

    i = -1
    for block_length in initial_state:
        i+=block_length
        indicies.append(i)

    return indicies[:-1]

# Measure the desired parameters at a time step in the simulation
def measurement(eng, data, target_state,final_model):
    keys = ['t', 'overlap', 'ene_exp']
    if data is None:
        data = dict([(k, []) for k in keys])
    data['t'].append(eng.evolved_time)
    #data['trunc_err'].append(eng.trunc_err.eps)
    data['ene_exp'].append(final_model.H_MPO.expectation_value(eng.psi))
    overlap_unsq = eng.psi.overlap(target_state)
    data['overlap'].append(overlap_unsq.conj()*overlap_unsq)
    return data

def get_measurement_times(time_samples):
    measurement_times = []
    for i in range(2**len(time_samples)):
        measurement_time = 0
        binary_string_coeffs = bin(i)[2:].rjust(len(time_samples), '0')
        for i in range(len(binary_string_coeffs)):
            measurement_time += int(binary_string_coeffs[i]) * time_samples[i]
        measurement_times.append(measurement_time)
    return sorted(measurement_times)

def rodeo_run_with_given_time_samples(hamiltonian, input_state, target_state, E_target, time_samples, tebd_params):
    time_evolution_engine = tebd.TEBDEngine(input_state.copy(), hamiltonian, tebd_params) 
    if len(time_samples) == 0:
        zero_input_state = input_state.copy()
        zero_norm_sq = zero_input_state.overlap(zero_input_state)
        zero_overlap = target_state.copy().overlap(zero_input_state)
        return zero_overlap.conj() * zero_overlap/zero_norm_sq, zero_norm_sq

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
    #print("finished time evolution")
   
    norm_sq = 0
    for i in range(len(coeffs_for_f_operators)):
        for j in range(len(coeffs_for_f_operators)):
            #print(coeffs_for_f_operators[i])
            #print(coeffs_for_f_operators[j])
            norm_sq += np.conj(coeffs_for_f_operators[i]) * coeffs_for_f_operators[j] * f_operators_without_coeffs[i].overlap(f_operators_without_coeffs[j])
    norm_sq /= (len(measurement_times)**2)
    #print("calculated all norms/overlaps")

    #print("!!!")
    #print(norm)

    #print(measurement_overlaps)
    #print([i.conj() * i for i in measurement_overlaps])
        #measurement(time_evolution_engine, data, psi_actual, final_model)
    overlap = np.sum(measurement_overlaps)/len(measurement_overlaps)
    #print(overlap)
    #print(overlap.conj()*overlap)
    return  overlap.conj() * overlap/norm_sq , norm_sq


    #overlap_unsq_after_ev = psi_actual.overlap(time_evolution_engine.psi) * 1/2 * np.exp(1j * E_targ * total_time)
    #overlap_unsq_after_ev += 1/2 * psi_actual.overlap(psi_i_for_i_overlap)
    #print("RODEO OVERLAP WITH FINAL")
    #print(overlap_unsq_after_ev.conj() * overlap_unsq_after_ev)

    #data["E0_uncoupled"] = E0_uncoupled
    #data["E0_coupled"] = E0_coupled

    #success_ov = 1/4 * psi_i_for_i_overlap.overlap(psi_i_for_i_overlap)
    #success_ov += 1/4 * time_evolution_engine.psi.overlap(time_evolution_engine.psi)
    #success_ov += 1/4 * np.exp(-1j * E_targ * total_time) * time_evolution_engine.psi.overlap(psi_i_for_i_overlap)
    #success_ov += 1/4 * np.exp(1j * E_targ * total_time) * psi_i_for_i_overlap.overlap(time_evolution_engine.psi)
    #success_chance = success_ov.conj() * success_ov

    #return overlap_unsq_after_ev.conj() * overlap_unsq_after_ev, success_chance

if __name__ == "__main__":
    from tqdm import tqdm

    # Model parameters
    J = -1
    SHAPE = [2]*2
    L = sum(SHAPE)
    mu = 0

    # Simulation parameters
    #E_target_vals = np.linspace(-1.1,-1,1)
    E_target_vals = np.linspace(-2,-0,15)
    sigma = 1
    #r = 3
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
    initial_ground_hamiltonian_dmrg = XXZChain({'Jxx':J*c_arr,"Jz":0,'hz':mu,"L":L})

    # Hamiltonian used for Rodeo Alg.
    rodeo_H = XXZChain({'Jxx':J,"Jz":0,'hz':mu,"L":L})

    # Guess for the ground state of the initial_model
    initial_state_guess = tenpy.networks.mps.MPS.from_lat_product_state(initial_ground_hamiltonian_dmrg.lat, [['up'], ['down']])

    # Use DMRG to calculate initial ground state of uncoupled model
    dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(initial_state_guess, initial_ground_hamiltonian_dmrg, dmrg_params)
    E0_uncoupled, initial_state = dmrg_eng_uncoupled_state.run()

    import new_model_tenpy as nmt
    AMI = nmt.AdiabaticHamiltonian({"Jxx": -4, "Jz":0, "hz":0, "L":L})
    AMF = XXZChain({'Jxx':J,"Jz":0,'hz':mu,"L":L})
    initial_state = nmt.complete_adiabatic_evolution_run(AMI, AMF, dmrg_params, tebd_params,1)

    # Take a copy of the initial state for later overlaps just in case it changes.
    #psi_i_for_i_overlap = initial_state.copy()

    # Get the exact ground state of the final Hamiltonian with DMRG
    # In reality we would not know this, but we use this to calculate overlaps
    dmrg_eng_final_state = dmrg.TwoSiteDMRGEngine(initial_state.copy(), rodeo_H, dmrg_params)
    E0_coupled, goal_state = dmrg_eng_final_state.run() 

    #print(goal_state.overlap(psi_i_for_i_overlap))
    print(goal_state.overlap(goal_state))

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
                RENAME_DATA,RENAME_DATA_2 = rodeo_run_with_given_time_samples(rodeo_H, initial_state, goal_state, E_target, time_samples, tebd_params)
                y_val_set.append(RENAME_DATA)
                y_val_set_2.append(RENAME_DATA_2)
            y_vals.append(y_val_set)
            y_vals_2.append(y_val_set_2)
        y_vals_FAKE.append(np.mean(y_vals,axis=0)[0])
        print(y_vals_FAKE)

        plt.plot(E_target_vals,np.mean(y_vals,axis=0),label="overlap")
        plt.plot(E_target_vals,np.mean(y_vals_2,axis=0),label="success_chance")
        plt.legend()
        plt.show()
    plt.plot(r_vals, y_vals_FAKE)
    plt.show()


    """
    resamples = 5
    y_vals = []
    y_vals2 = []
    time_samples = np.abs(sigma*np.random.randn(resamples))
        y_val_set = []
        y_val_set2 = []
        for total_time in tqdm(time_samples,leave=False): #sigma*np.random.randn(resamples):
            #print("TOTAL_TIME:")
            #print(total_time)
            gstate_overlap, success_prob = complete_rodeo_run(initial_ground_hamiltonian_dmrg, rodeo_H, dmrg_params, tebd_params, E_t, total_time, verbose=False)
            y_val_set.append(gstate_overlap)
            y_val_set2.append(success_prob)
        y_vals.append(np.average(y_val_set))
        y_vals2.append(np.average(y_val_set2))
    plt.plot(E_t_vals,y_vals,label="overlap")
    plt.plot(E_t_vals,y_vals2,label="success chance")
    plt.legend()
    plt.show()
    """

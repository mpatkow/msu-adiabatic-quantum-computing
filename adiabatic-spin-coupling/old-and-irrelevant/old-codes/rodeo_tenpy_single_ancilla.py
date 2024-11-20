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

def get_intermediate_measurement_times(tsamples):
    for 

# Run a complete adiabatic evolution (single run) from the initial_model to the final_model
def complete_rodeo_run(initial_model, final_model, dmrg_params, tebd_params, E_targ, total_time, verbose=True):
    # Guess for the ground state of the initial_model
    psi0_i_guess = tenpy.networks.mps.MPS.from_lat_product_state(initial_model.lat, [['up'], ['down']])

    dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(psi0_i_guess, initial_model, dmrg_params)
    E0_uncoupled, psi_start = dmrg_eng_uncoupled_state.run()
    psi_i_for_i_overlap = psi_start.copy()

    if verbose:
        print(f"Ground state of initial model: {E0_uncoupled}")

    dmrg_eng_final_state = dmrg.TwoSiteDMRGEngine(psi_start.copy(), final_model, dmrg_params)
    E0_coupled, psi_actual = dmrg_eng_final_state.run() 

    if verbose:
        print(f"Ground state of final model: {E0_coupled}")
        print("\nDMRG step finished\n\n======================================================\nTime Evolution Preparation...\n")

    #time_evolution_engine = tebd.TEBDEngine(psi_start, initial_model, tebd_params) 
    #time_evolution_engine = tebd.TEBDEngine(psi_actual.copy(), final_model, tebd_params) 
    time_evolution_engine = tebd.TEBDEngine(psi_start.copy(), final_model, tebd_params) 

    #data = measurement(time_evolution_engine, None, psi_actual, final_model)

    if verbose:
        print("Time Evolution Running...")

    print("TOTAL_TIME: "+ str(total_time))
    while time_evolution_engine.evolved_time < total_time:
        print("TIME: " + str(time_evolution_engine.evolved_time))
        if verbose:
            print(f" Time Evolution step is {time_evolution_engine.evolved_time/total_time * 100}% complete.")
        time_evolution_engine.run()
        
        #measurement(time_evolution_engine, data, psi_actual, final_model)

        #initial_model.init_H_from_terms()
        #initial_model.update_time_parameter(time_evolution_engine.evolved_time)

    overlap_unsq_after_ev = psi_actual.overlap(time_evolution_engine.psi) * 1/2 * np.exp(1j * E_targ * total_time)
    overlap_unsq_after_ev += 1/2 * psi_actual.overlap(psi_i_for_i_overlap)
    #print("RODEO OVERLAP WITH FINAL")
    #print(overlap_unsq_after_ev.conj() * overlap_unsq_after_ev)

    #data["E0_uncoupled"] = E0_uncoupled
    #data["E0_coupled"] = E0_coupled

    success_ov = 1/4 * psi_i_for_i_overlap.overlap(psi_i_for_i_overlap)
    success_ov += 1/4 * time_evolution_engine.psi.overlap(time_evolution_engine.psi)
    success_ov += 1/4 * np.exp(-1j * E_targ * total_time) * time_evolution_engine.psi.overlap(psi_i_for_i_overlap)
    success_ov += 1/4 * np.exp(1j * E_targ * total_time) * psi_i_for_i_overlap.overlap(time_evolution_engine.psi)
    success_chance = success_ov.conj() * success_ov

    return overlap_unsq_after_ev.conj() * overlap_unsq_after_ev, success_chance

if __name__ == "__main__":
    from tqdm import tqdm

    # Simulation parameters
    h = 0
    TOTAL_TIME = 20
    J = -1
    SHAPE = [2]*2
    #SHAPE_F = [16]
    L = sum(SHAPE)
    total_runtimes = np.linspace(1,5,5)
    EPSILON_RODEO = 0.1
   
    from tenpy.models.xxz_chain import XXZChain

    mu = 0
    #new_hamiltonian = AdiabaticHamiltonian({"Jxx":J, "Jz":0, "hz":mu, "L":L})
    #Z_operator = XXZChain({'Jxx':0,"Jz":0,'hz':1,"L":L}).calc_H_MPO()

    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-10,
        },
        'combine': True,
    }

    #psi0_i_guess = tenpy.networks.mps.MPS.from_lat_product_state(new_hamiltonian.lat, [['down'],['up']])


    #print(f"<Z> guess {Z_operator.expectation_value(psi0_i_guess)}")
    #print(f"<E> guess {new_hamiltonian.calc_H_MPO().expectation_value(psi0_i_guess)}")
    #dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(psi0_i_guess, new_hamiltonian, dmrg_params)
    #E0_uncoupled, psi_start = dmrg_eng_uncoupled_state.run()
    #print("=== DMRG ... ===")
    #print(f"<Z> ground {Z_operator.expectation_value(psi_start)}")
    #print(f"<E> ground {E0_uncoupled}")


    c_arr = np.ones(L-1)
    for non_coupling_index in get_non_interaction_term_indicies(SHAPE):
        c_arr[non_coupling_index] = 0
    initial_ground_hamiltonian_dmrg = XXZChain({'Jxx':J*c_arr,"Jz":0,'hz':mu,"L":L})
    rodeo_H = XXZChain({'Jxx':J,"Jz":0,'hz':mu,"L":L})
    #M_f = XXZChain({'J':J*c_arr,'g':h,"L":L})

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

    E_t_vals = np.linspace(-4,3,20)
    sigma = 1
    resamples = 5
    y_vals = []
    y_vals2 = []
    time_samples = np.abs(sigma*np.random.randn(resamples))
    import tqdm
    for E_t in tqdm.tqdm (E_t_vals):
        y_val_set = []
        y_val_set2 = []
        for total_time in tqdm.tqdm(time_samples,leave=False): #sigma*np.random.randn(resamples):
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


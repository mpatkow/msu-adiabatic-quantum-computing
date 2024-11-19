# Most recent code as of August 12 2024 to adiabatically
# evolve XX chains with tenpy

import numpy as np
import tenpy
from tenpy.algorithms import dmrg, tebd
from tenpy.models.xxz_chain import XXZChain2 as XXZChain
from tenpy.models.hubbard import FermiHubbardChain

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

# Run a complete adiabatic evolution (single run) from the initial_model to the final_model
def complete_adiabatic_evolution_run(initial_model, final_model, ground_state_guess, dmrg_params, tebd_params, total_time, verbose=True, initial_state=None):
    # Guess for the ground state of the initial_model
    psi0_i_guess = tenpy.networks.mps.MPS.from_lat_product_state(initial_model.lat,ground_state_guess)

    dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(psi0_i_guess, initial_model, dmrg_params)
    E0_uncoupled, psi_start = dmrg_eng_uncoupled_state.run()

    if verbose:
        print(f"Ground state of initial model: {E0_uncoupled}")

    dmrg_eng_final_state = dmrg.TwoSiteDMRGEngine(psi_start.copy(), final_model, dmrg_params)
    E0_coupled, psi_actual = dmrg_eng_final_state.run() 

    if verbose:
        print(f"Ground state of final model: {E0_coupled}")

    if verbose:
        print("\nDMRG step finished\n\n======================================================\nTime Evolution Preparation...\n")
    
    if initial_state != None:
        psi_start = initial_state.copy()

    time_evolution_engine = tebd.TimeDependentTEBD(psi_start, initial_model, tebd_params) 

    data = measurement(time_evolution_engine, None, psi_actual, final_model)

    if verbose:
        print("Time Evolution Running...")

    while time_evolution_engine.evolved_time < total_time:
        if verbose:
            print(f" Time Evolution step is {time_evolution_engine.evolved_time/total_time * 100}% complete.")
        time_evolution_engine.run()
        
        measurement(time_evolution_engine, data, psi_actual, final_model)

        initial_model.init_H_from_terms()
        initial_model.update_time_parameter(time_evolution_engine.evolved_time)

    data["E0_uncoupled"] = E0_uncoupled
    data["E0_coupled"] = E0_coupled

    return data, time_evolution_engine.psi



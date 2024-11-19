import numpy as np
import tenpy
import matplotlib.pyplot as plt
from tenpy.algorithms import dmrg, tebd
from tenpy.models.xxz_chain import XXZChain2 as XXZChain
import rodeo_tenpy as rt
from tqdm import tqdm
import hubbard_hamiltonians as hubhams
from tenpy.models.hubbard import FermiHubbardChain

def adiabatic_rodeo_run(ar_params):
    # Construct initial hamiltonian with missing coupling terms
    M_i = hubhams.AdiabaticHubbardHamiltonian(ar_params['special_xxzmodel_params'] | {"adiabatic_time":ar_params['adiabatic_params']['adiabatic_time']})

    # Final/Target Hamiltonian
    M_f = FermiHubbardChain(ar_params['special_xxzmodel_params'])

    # Get initial state from adiabatic method
    if ar_params["saved_mps_bool"]:
        import pickle
        with open(ar_params["saved_mps_filename"], 'rb') as f:
            initial_state = pickle.load(f)
    else:
        import adiabatic_fusion_tenpy_xxchain as aft
        run_data, initial_state = aft.complete_adiabatic_evolution_run(M_i, M_f, ar_params['initial_guess_lat'], ar_params['dmrg_params'], ar_params['adiabatic_params']['tebd_params'], ar_params['adiabatic_params']['adiabatic_time'], verbose=ar_params['adiabatic_params']['verbose'])
    if ar_params["save_mps_bool"]:
        import pickle
        with open(ar_params["saved_mps_filename"], 'wb') as f:
            pickle.dump(initial_state, f)
    # Guess for the ground state of the initial_model
    #initial_state_guess = tenpy.networks.mps.MPS.from_lat_product_state(M_i.lat, ar_params['initial_guess_lat'])
    #run_data, initial_state = aft.complete_adiabatic_evolution_run(M_i, M_f, initial_guess_lat, dmrg_params, tebd_params, ADIABATIC_TIME, verbose=False, initial_state=initial_state_guess)

    # Get the exact ground state of the final Hamiltonian with DMRG
    # In reality we would not know this, but we use this to calculate overlaps
    dmrg_eng_final_state = dmrg.TwoSiteDMRGEngine(initial_state.copy(), M_f, ar_params['dmrg_params'])
    E0_coupled, goal_state = dmrg_eng_final_state.run() 

    initial_ov = initial_state.overlap(goal_state)
    print(f"Initial Overlap (lower bound on success probability): {initial_ov * initial_ov.conj()}")

    overlap_values = []
    success_probability_values = []
    time_samples = []
    for resample_i in tqdm(range(ar_params['rodeo_params']['resamples'])):
        if not ar_params["super_iterations"]:
            time_samples = np.abs(ar_params['rodeo_params']['sigma']*np.random.randn(ar_params['rodeo_params']['r_value'])) # Absolute value as TEBD doesn't work with negative times
            time_samples = [time_sample // ar_params['rodeo_params']['tebd_params']['dt'] * ar_params['rodeo_params']['tebd_params']['dt'] for time_sample in time_samples]
        else:
            time_samples = ar_params['rodeo_params']['r_value']*[ar_params['rodeo_params']['sigma'] * 2 ** (-exponent) for exponent in range(ar_params['rodeo_params']['r_superiteration'])]
            time_samples = [time_sample // ar_params['rodeo_params']['tebd_params']['dt'] * ar_params['rodeo_params']['tebd_params']['dt'] for time_sample in time_samples]
        if ar_params['verbose']:
            print(time_samples)
        if ar_params["use_dmrg_etarget"]:
            print(E0_coupled)
            overlap_value, success_probability_value = rt.rodeo_run(M_f, initial_state, goal_state, E0_coupled, time_samples, ar_params['rodeo_params']['tebd_params'])
        else:
            overlap_value, success_probability_value = rt.rodeo_run(M_f, initial_state, goal_state, ar_params['rodeo_params']['e_target'], time_samples, ar_params['rodeo_params']['tebd_params'])
        overlap_values.append(overlap_value)
        if ar_params['verbose']:
            print("success_probability_value")
            print(success_probability_value)
            print("overlap_value:")
            print(overlap_value)
        success_probability_values.append(success_probability_value)

    calculated_overlap = np.mean(overlap_values)
    calculated_success_probability = np.mean(success_probability_values)
    #calculated_cost = ar_params['adiabatic_params']['adiabatic_time'] if not ar_params['rodeo_params']['r_value'] else 1/calculated_success_probability * (ar_params['adiabatic_params']['adiabatic_time'] + ar_params['rodeo_params']['r_value'] * ar_params['rodeo_params']['sigma'])
    calculated_cost = ar_params['adiabatic_params']['adiabatic_time'] if not ar_params['rodeo_params']['r_value'] else 1/calculated_success_probability * (ar_params['adiabatic_params']['adiabatic_time'] + np.sum(time_samples))

    if ar_params['verbose']: print(f'Overlap: {calculated_overlap.real}, success chance: {calculated_success_probability.real}, cost: {calculated_cost.real}.')

    if ar_params['write_to_outfile']:
        with open(ar_params['outfile'],"w") as f:
            header_to_write = ['overlap','success_probability','cost']
            values_to_write = [str(res_value.real) for res_value in [calculated_overlap, calculated_success_probability, calculated_cost]]
            to_write_string = ",".join(header_to_write) + "\n" + ",".join(values_to_write) + "\n"

            f.write(to_write_string)

    return calculated_overlap, calculated_success_probability, calculated_cost


def main():
    import json
    import sys

    adiabatic_rodeo_parameters_filename = sys.argv[1] 

    with open(adiabatic_rodeo_parameters_filename) as f:
        adiabatic_rodeo_parameters = json.load(f)

    co, csp, cc = adiabatic_rodeo_run(adiabatic_rodeo_parameters)
    
    with open(adiabatic_rodeo_parameters_filename, "a") as f:
        header_to_write = ['overlap','success_probability','cost']
        values_to_write = [str(res_value.real) for res_value in [co, csp, cc]]
        to_write_string = "\n" + ",".join(header_to_write) + "\n" + ",".join(values_to_write) + "\n"

        f.write(to_write_string)



if __name__ == "__main__":
    main() 

# Attempt to create and adiabatically evolve Ising chains in tenpy

import numpy as np
import scipy
import tenpy
import matplotlib.pyplot as plt
from tenpy.algorithms import dmrg, tebd
from tenpy.models.tf_ising import TFIChain

# Return bound indicies that do not have an interaction initially, based on original chain.
def get_non_interaction_term_indicies(initial_state):
    indicies = []

    i = -1
    for block_length in initial_state:
        i+=block_length
        indicies.append(i)

    return indicies[:-1]

# Model for the transverse ising chain that linearly interpolates the coupling terms based on shape provided
class AdiabaticHamiltonian(TFIChain):
    def init_terms(self, model_params):
        c_arr = np.ones(L-1)
        for non_coupling_index in get_non_interaction_term_indicies(SHAPE):
            c_arr[non_coupling_index] = model_params.get('time', 0)/TOTAL_TIME

        model_params['J'] = c_arr * J

        super().init_terms(model_params)

# Measure the desired parameters at a time step in the simulation
def measurement(eng, data, target_state, final_model):
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
def complete_adiabatic_evolution_run(initial_model, final_model, dmrg_params, tebd_params, total_time, verbose=True):
    # Guess for the ground state of the initial_model
    psi0_i_guess = tenpy.networks.mps.MPS.from_lat_product_state(initial_model.lat, [['up']])

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

    time_evolution_engine = tebd.TimeDependentTEBD(psi_start, initial_model, tebd_params) 
    #time_evolution_engine = tebd.TEBDEngine(psi_start, M_i, tebd_params) 

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
    return data

if __name__ == "__main__":
    from tqdm import tqdm

    # Simulation parameters
    h = 0.5
    TOTAL_TIME = 10
    J = -1
    SHAPE = [2]*2
    L = sum(SHAPE)
    total_runtimes = [10]
    EPSILON_RODEO = 0.1
    
    M_i = AdiabaticHamiltonian({'J':J,'g':h,"L":L})
    M_f = TFIChain({'J':J,'g':h,"L":L})

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
        'dt': 1,
        'order': 4,
        'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
    }

    data = {'total_runtimes':total_runtimes, 'overlap_at_end':[], 'estimated_cost_adiabatic_rodeo':[], 'estimated_cost_rodeo_only':[], 'estimated_cost_adiabatic_rodeo_2':[], 'estimated_cost_rodeo_only_2':[]}

    x_plots = []
    y_plots = []
    for TOTAL_TIME in tqdm(total_runtimes):
        run_data = complete_adiabatic_evolution_run(M_i, M_f, dmrg_params, tebd_params, TOTAL_TIME)
        plt.plot(run_data['t'], run_data['overlap'])
        plt.show()
        x_plots.append(run_data['t'])
        y_plots.append(run_data['overlap'])
        data['overlap_at_end'].append(run_data['overlap'][-1])

        # Calculate the estimated cost. That is 1/overlap * adiabatic_time
        data['estimated_cost_adiabatic_rodeo'].append(run_data['t'][-1] * 1/run_data['overlap'][-1])
        #print(f"Estimated cost for applying rodeo after a single [2]*n -> [2n] adiabatic fusion: {estimated_cost_adiabatic_rodeo}")

        # Calculate the estimated cost for only rodeo. That is 1/(overlap (t=0))
        data['estimated_cost_rodeo_only'].append(1/run_data['overlap'][0])
        #print(f"Estimated cost for applying rodeo to initial state of [2]*n: {estimated_cost_rodeo_only}")

        # Calculate the estimated cost by new method.  
        a_sq_end = run_data['overlap'][-1]
        N_rodeo_end = max(1,np.log2(1/EPSILON_RODEO * (1/a_sq_end - 1)))
        a_sq_start = run_data['overlap'][0]
        N_rodeo_start = max(1,np.log2(1/EPSILON_RODEO * (1/a_sq_start - 1)))
        data['estimated_cost_adiabatic_rodeo_2'].append(run_data['t'][-1] * N_rodeo_end/a_sq_end)
        data['estimated_cost_rodeo_only_2'].append(N_rodeo_start/a_sq_start)


    plt.subplot(1,2,1)
    plt.plot(data['total_runtimes'], data['overlap_at_end'], color="black", linestyle="dashed")
    #for x_plot,y_plot in zip(x_plots, y_plots):
    #    plt.plot(x_plot, y_plot)
    plt.xlabel(r"Total runtime $T$")
    plt.ylabel(r"Overlap $|\langle \psi _0 | \phi \rangle |^2$")
    plt.subplot(1,2,2)
    plt.plot(data['total_runtimes'], np.divide(data['estimated_cost_adiabatic_rodeo'], data['estimated_cost_rodeo_only']), label="original_method")
    plt.plot(data['total_runtimes'], np.divide(data['estimated_cost_adiabatic_rodeo_2'], data['estimated_cost_rodeo_only_2']), label="including rodeo cycles")
    plt.legend()
    plt.xlabel(r"Total runtime $T$")
    plt.ylabel(r"Adiabatic Rodeo Cost / Rodeo Only Cost")
    plt.show()


    def exp_fit_overlap(x, a, b):
        return 1-a*np.exp(x*b) 

    x = data['total_runtimes']
    yn = data['overlap_at_end']

    popt, pcov = scipy.optimize.curve_fit(exp_fit_overlap, x, yn, p0=(5,-0.5))

    plt.plot(x, yn, 'ko', label="Original Noised Data")
    plt.plot(x, exp_fit_overlap(x, *popt), 'r-', label="Fitted Curve")
    plt.legend()
    plt.show()

    #plt.subplot(1,2,1)
    #plt.plot(data['t'], data['overlap'])
    #plt.xlabel(r"Evolution time $t$")
    #plt.ylabel(r"Overlap $|\langle \psi _0 | \phi \rangle |^2$")
    #plt.hlines(1, plt.xlim()[0], plt.xlim()[1],color="black",linestyle="dashed")
    #plt.ylim(-0.1, 1.1)
    #plt.subplot(1,2,2)
    #plt.plot(data['t'], data['ene_exp'])
    #plt.hlines(E, plt.xlim()[0], plt.xlim()[1],color="black",linestyle="dashed")
    #plt.hlines(E2, plt.xlim()[0], plt.xlim()[1],color="black",linestyle="dashed")
    #plt.xlabel(r"Evolution time $t$")
    #plt.ylabel(r"Energy Expectation Value $\langle \psi | H_f |\psi \rangle$")
    #plt.show()

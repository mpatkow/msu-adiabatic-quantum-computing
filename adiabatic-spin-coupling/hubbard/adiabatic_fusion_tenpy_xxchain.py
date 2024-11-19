# Most recent code as of August 12 2024 to adiabatically
# evolve XX chains with tenpy

import numpy as np
import tenpy
from tenpy.algorithms import dmrg, tebd
from tenpy.models.xxz_chain import XXZChain2 as XXZChain

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


def main():
    from tqdm import tqdm
    import special_xxz_hamiltonians as xxzhs

    # Simulation parameters
    h = 0
    J = 1 # Usually 1
    SHAPE = [4]*2
    #SHAPE2 = [2]*2
    #TOTAL_TIME = 20
    #SHAPE_F = [16]
    L = sum(SHAPE)
    #L2 = sum(SHAPE2)
    #total_runtimes = np.linspace(6.45,6.55,20)
    #total_runtimes = np.linspace(7.8,7.9,10)
    EPSILON_RODEO = 0.1
    Jz = -0.4       # -0.9 #-2 #-10 #-1.5 #-0.5
    bp_h = 0 #-0.1 #-0.1 #-0.001 #-1e-1 #-0.01 #-0.000001 #-1
   
    #from tenpy.models.xxz_chain import XXZChain

    epsilon = 0.01
    mu = 0
    #new_hamiltonian = AdiabaticHamiltonian({"Jxx":J, "Jz":Jz, "hz":mu, "L":L})
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

    #psi0_i_guess = tenpy.networks.mps.MPS.from_lat_product_state(new_hamiltonian.lat, [['up'],['down']])


    #print(f"<Z> guess {Z_operator.expectation_value(psi0_i_guess)}")
    #print(f"<E> guess {new_hamiltonian.calc_H_MPO().expectation_value(psi0_i_guess)}")
    #dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(psi0_i_guess, new_hamiltonian, dmrg_params)
    #E0_uncoupled, psi_start = dmrg_eng_uncoupled_state.run()
    #print("=== DMRG ... ===")
    #print(f"<Z> ground {Z_operator.expectation_value(psi_start)}")
    #print(f"<E> ground {E0_uncoupled}")

    M_f = xxzhs.XXZChainWithCenterPotential({'Jxx':J,"Jz":Jz,'hz':mu,"L":L, "boundary_potential":bp_h})
    #M_f2 = XXZChain({'Jxx':J,"Jz":Jz,'hz':mu,"L":L2})
        #M_i = AdiabaticHamiltonian({"Jxx":J, "Jz":Jz, "hz":mu, "L":L})
    #c_arr = np.ones(L-1)
    #for non_coupling_index in get_non_interaction_term_indicies(SHAPE_F):
    #    c_arr[non_coupling_index] = 0
    #M_f = XXZChain({'Jxx':J,"Jz":Jz,'hz':mu,"L":L})
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
        'dt': 0.5,
        'order': 4,
        'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
    }

    prec = 1
    total_runtimes = np.linspace(0,2,2)
    step = 2
    while True:
        data = {'total_runtimes':total_runtimes, 'overlap_at_end':[], 'estimated_cost_adiabatic_rodeo':[], 'estimated_cost_rodeo_only':[], 'estimated_cost_adiabatic_rodeo_2':[], 'estimated_cost_rodeo_only_2':[]}

        x_plots = []
        y_plots = []
        for TOTAL_TIME in tqdm(total_runtimes):
            M_i = xxzhs.AdiabaticHamiltonianWithCenterPotential({"Jxx":J, "Jz":Jz, "hz":mu, "L":L, "shape":SHAPE, "adiabatic_time":TOTAL_TIME, "Jxx_coeff":J, "Jz_coeff":Jz, "boundary_potential":bp_h})
            #M_i = AdiabaticHamiltonian({"Jxx":J, "Jz":Jz, "hz":mu, "L":L, "shape":SHAPE, "adiabatic_time":TOTAL_TIME, "Jxx_coeff":J, "Jz_coeff":Jz})
            #M_i2 = xxzhs.AdiabaticHamiltonian({"Jxx":J, "Jz":Jz, "hz":mu, "L":L2, "shape":SHAPE, "adiabatic_time":TOTAL_TIME})
            #run_data1, psi_adiabatic_result_22 = complete_adiabatic_evolution_run(M_i2, M_f2, dmrg_params, tebd_params, TOTAL_TIME)
            #combined_initial_state = tenpy.networks.mps.MPS.from_product_mps_covering([psi_adiabatic_result_22,psi_adiabatic_result_22],[[0,1,2,3],[4,5,6,7]])
            #LAT_PROD = [['down'],['down'],['up'],['up']]  # [['up'],['down']]   #[['up'],['up'],['down'],['down']]
            LAT_PROD = [['down'],['up']]
            #LAT_PROD = [['up'],['up'],['up'],['up'],['down'],['down'],['down'],['down']]
            DIRECT_INPUT_STATE = tenpy.networks.mps.MPS.from_lat_product_state(M_i.lat,LAT_PROD)
            #run_data, _ = complete_adiabatic_evolution_run(M_i, M_f, [["up"],["down"]], dmrg_params, tebd_params, TOTAL_TIME) #initial_state = psi_adiabatic_result_)
            run_data, _ = complete_adiabatic_evolution_run(M_i, M_f, LAT_PROD, dmrg_params, tebd_params, TOTAL_TIME) #, initial_state = DIRECT_INPUT_STATE)
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
        if np.max(run_data['overlap']) > 1 - epsilon:
            prec *= 0.1
            total_runtimes = np.linspace(int(np.min(total_runtimes)),int(np.max(total_runtimes)), int((np.max(total_runtimes)-np.min(total_runtimes))/prec)+1)
        else:
            total_runtimes = np.linspace(int(np.max(total_runtimes)), int(np.max(total_runtimes))+step, int(step/prec)+1)
        if prec < 0.02:
            break

        print("!!!!!!!!!!")
        print(total_runtimes)
        print("!!!!!!!!!!")

    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    #plt.plot(data['total_runtimes'], np.ones(len(data['overlap_at_end']))-data['overlap_at_end'], color="black", linestyle="dashed")
    plt.plot(data['total_runtimes'], data['overlap_at_end'], color="black", linestyle="dashed")
    #plt.yscale("log")
    #for x_plot,y_plot in zip(x_plots, y_plots):
    #    plt.plot(x_plot, y_plot)
    plt.xlabel(r"Total runtime $T$")
    #plt.ylabel(r"1 minus Overlap $1-|\langle \psi _0 | \phi \rangle |^2$")
    plt.ylabel(r"Overlap $|\langle \psi _0 | \phi \rangle |^2$")
    plt.subplot(1,2,2)
    plt.plot(data['total_runtimes'], np.divide(data['estimated_cost_adiabatic_rodeo'], data['estimated_cost_rodeo_only']), label="original_method")
    plt.plot(data['total_runtimes'], np.divide(data['estimated_cost_adiabatic_rodeo_2'], data['estimated_cost_rodeo_only_2']), label="including rodeo cycles")
    plt.legend()
    plt.xlabel(r"Total runtime $T$")
    plt.ylabel(r"Adiabatic Rodeo Cost / Rodeo Only Cost")
    plt.show()

    time_values = data['total_runtimes']
    overlap_values = data['overlap_at_end']

    overlap_values_epsilon_boolean_i = [overlap_value > 1-epsilon for overlap_value in overlap_values].index(np.True_)
    print(f"minimum time when overlap is within {epsilon} of 1: {time_values[overlap_values_epsilon_boolean_i]}")

    exit()

    outfile = "adiabatic_result.dat"
    with open(outfile, "a") as f:
        firstline = ""
        for keyname in data.keys():
            firstline+=keyname
            firstline+=","
        firstline = firstline[:-1]
        f.write(firstline+"\n")
        things_to_write = []
        for keyname in data.keys():
            things_to_write.append(data[keyname])
        for i in range(len(things_to_write[0])):
            to_write = ""
            for thing in things_to_write:
                to_write += str(float(thing[i]))
                to_write += ","
            to_write = to_write[:-1]
            f.write(to_write +"\n")

    def exp_fit_overlap(x, a, b):
        return 1-a*np.exp(x*b) 

    x = data['total_runtimes']
    yn = data['overlap_at_end']

    import scipy
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

if __name__ == "__main__":
    main()

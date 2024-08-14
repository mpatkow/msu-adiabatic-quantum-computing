import numpy as np
import tenpy
import matplotlib.pyplot as plt
from tenpy.algorithms import dmrg, tebd
from tenpy.models.xxz_chain import XXZChain2 as XXZChain
import rodeo_tenpy as rt
from tqdm import tqdm

# Model parameters
J = -1
#SHAPE = [8]*2
SHAPE = [2]*2
L = sum(SHAPE)
mu = 0

# Simulation parameters
#E_target_vals = np.linspace(-1.120,-1.100,8)
#E_target_vals = np.linspace(-1.15,-1.05,10)
E_target_vals = np.linspace(-1.1180,-1.1179,1)
#E_target_vals = np.linspace(-4.91898,-4.91898,2)
#E_target_vals = np.linspace(-4.95,-4.9,5)
#E_target_vals = np.linspace(-1.5,-0.5,20)
sigma = 20
#r = 3
r_vals = [6]
#r_vals = [0] 
resamples = 50
ADIABATIC_TIME = 0

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
for non_coupling_index in rt.get_non_interaction_term_indicies(SHAPE):
    c_arr[non_coupling_index] = 0
initial_ground_hamiltonian_dmrg = XXZChain({'Jxx':J*c_arr,"Jz":0,'hz':mu,"L":L})

# Hamiltonian used for Rodeo Alg.
rodeo_H = XXZChain({'Jxx':J,"Jz":0,'hz':mu,"L":L})

# Guess for the ground state of the initial_model
initial_state_guess = tenpy.networks.mps.MPS.from_lat_product_state(initial_ground_hamiltonian_dmrg.lat, [['up'], ['down']])

# Use DMRG to calculate initial ground state of uncoupled model
#dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(initial_state_guess, initial_ground_hamiltonian_dmrg, dmrg_params)
#dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(initial_state_guess, rodeo_H, dmrg_params)
#E0_uncoupled, initial_state = dmrg_eng_uncoupled_state.run()

# Get initial state from adiabatic method

class AdiabaticHamiltonian(XXZChain):
    def init_terms(self, model_params):
        c_arr = np.ones(L-1)
        for non_coupling_index in rt.get_non_interaction_term_indicies(SHAPE):
            if ADIABATIC_TIME != 0:
                c_arr[non_coupling_index] = model_params.get('time', 0)/ADIABATIC_TIME
            else:
                c_arr[non_coupling_index] = 0

        model_params['Jxx'] = c_arr * J
        print("!!!")
        print(model_params['Jxx'])

        super().init_terms(model_params)


import adiabatic_fusion_tenpy_xxchain as aft
M_i = AdiabaticHamiltonian({"Jxx":J, "Jz":0, "hz":mu, "L":L})
M_f = XXZChain({'Jxx':J,"Jz":0,'hz':mu,"L":L})
run_data, initial_state = aft.complete_adiabatic_evolution_run(M_i, M_f, dmrg_params, tebd_params, ADIABATIC_TIME)

# Get the exact ground state of the final Hamiltonian with DMRG
# In reality we would not know this, but we use this to calculate overlaps
dmrg_eng_final_state = dmrg.TwoSiteDMRGEngine(initial_state.copy(), rodeo_H, dmrg_params)
E0_coupled, goal_state = dmrg_eng_final_state.run() 

#print(goal_state.overlap(psi_i_for_i_overlap))

y_vals_FAKE = []
for r in r_vals:
    y_vals = []
    y_vals_2 = []
    #time_samples = np.abs(sigma*np.random.randn(r)) # Absolute value as TEBD doesn't work with negative times
    for resample_i in tqdm(range(resamples)):
        y_val_set = []
        y_val_set_2 = []
        time_samples = np.abs(sigma*np.random.randn(r)) # Absolute value as TEBD doesn't work with negative times
        #print("TIME SAMPLES")
        #print(time_samples)
        for E_target in tqdm(E_target_vals,leave=False):
            #time_samples = np.abs(sigma*np.random.randn(r)) # Absolute value as TEBD doesn't work with negative times
            RENAME_DATA,RENAME_DATA_2 = rt.rodeo_run_with_given_time_samples(rodeo_H, initial_state, goal_state, E_target, time_samples, tebd_params)
            y_val_set.append(RENAME_DATA)
            y_val_set_2.append(RENAME_DATA_2)
        y_vals.append(y_val_set)
        y_vals_2.append(y_val_set_2)
    y_vals_FAKE.append(np.mean(y_vals,axis=0)[0])
    print('overlap')
    print(np.mean(y_vals,axis=0))
    OVERLAP_TO_WRITE = str(np.mean(y_vals,axis=0)[0].real)
    calculated_overlap = np.mean(y_vals,axis=0)[0]
    #print("standard deviation")
    #iiiii = 0
    #while True:
    #    try:
    #        print(np.std(np.array(y_vals)[:,iiiii]))
    #        iiiii+=1
    #    except:
    #        break
    #        
    calculated_sp = np.mean(y_vals_2,axis=0)[0]
    print('success chance')
    print(np.mean(y_vals_2,axis=0))
    SUCCESS_CHANCE_TO_WRITE = str(np.mean(y_vals_2,axis=0)[0].real)
    print('cost')
    if r == 0:
        print(ADIABATIC_TIME)
    else:
        print(r/calculated_sp*(ADIABATIC_TIME + sigma))
    #print("standard deviation")
    #iiiii = 0
    #while True:
    #    try:
    #        print(np.std(np.array(y_vals_2)[:,iiiii]))
    #        iiiii+=1
    #    except:
    #        break
    plt.plot(E_target_vals,np.mean(y_vals,axis=0),label="overlap")
    #for yvalset in y_vals:
        #plt.scatter(E_target_vals,yvalset,label="overlap")
    plt.plot(E_target_vals,np.mean(y_vals_2,axis=0),label="success_chance")
    #for yvalset2 in y_vals_2:
    #    plt.scatter(E_target_vals,yvalset2,label="success_chance")
    plt.legend()
    plt.show()

    TOWRITENAME = "adiabatic_rodeo_results.dat"
    with open(TOWRITENAME,"a") as f:
        towritestring = ""
        for shapev in SHAPE:
            towritestring += str(shapev)
        towritestring += ','
        towritestring += str(sigma) + "," + str(r) + "," + str(ADIABATIC_TIME) + "," + str(resamples) + "," + OVERLAP_TO_WRITE + "," + SUCCESS_CHANCE_TO_WRITE + "," + str(tebd_params['dt']) + "\n"
        f.write(towritestring)


plt.plot(r_vals, y_vals_FAKE)
plt.show()

####################################################################################

"""

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
"""

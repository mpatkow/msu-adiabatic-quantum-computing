import numpy as np
import tenpy
import matplotlib.pyplot as plt
from tenpy.algorithms import dmrg, tebd
from tenpy.models.xxz_chain import XXZChain2 as XXZChain
import rodeo_tenpy as rt
from tqdm import tqdm
import special_xxz_hamiltonians as xxzhs

def run(J, Jz, SHAPE, mu, E_target_vals, sigma, r_vals, resamples, ADIABATIC_TIME):
    L = sum(SHAPE)
    #bp_h = 0.0000001
    bp_h = 0 #-0.1
    #initial_guess_lat = [['up'],['up'],['down'],['down']]
    initial_guess_lat = [['down'],['up']]

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
    initial_ground_hamiltonian_dmrg = xxzhs.AdiabaticHamiltonianWithCenterPotential({'Jxx':J,"Jz":Jz,'hz':mu,"L":L,"shape":SHAPE,"adiabatic_time":ADIABATIC_TIME, "boundary_potential":bp_h,"Jxx_coeff":J,"Jz_coeff":Jz})
    #initial_ground_hamiltonian_dmrg = xxzhs.AdiabaticHamiltonian({'Jxx':J,"Jz":Jz,'hz':mu,"L":L,"shape":SHAPE,"adiabatic_time":ADIABATIC_TIME,"Jxx_coeff":J,"Jz_coeff":Jz})

    # Hamiltonian used for Rodeo Alg.
    rodeo_H = xxzhs.XXZChainWithCenterPotential({'Jxx':J,"Jz":Jz,'hz':mu,"L":L,"boundary_potential":bp_h})

    # Guess for the ground state of the initial_model
    initial_state_guess = tenpy.networks.mps.MPS.from_lat_product_state(initial_ground_hamiltonian_dmrg.lat, initial_guess_lat)

    # Use DMRG to calculate initial ground state of uncoupled model
    #dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(initial_state_guess, initial_ground_hamiltonian_dmrg, dmrg_params)
    #dmrg_eng_uncoupled_state = dmrg.TwoSiteDMRGEngine(initial_state_guess, rodeo_H, dmrg_params)
    #E0_uncoupled, initial_state = dmrg_eng_uncoupled_state.run()

    # Get initial state from adiabatic method
    import adiabatic_fusion_tenpy_xxchain as aft
    M_i = xxzhs.AdiabaticHamiltonianWithCenterPotential({"Jxx":J, "Jz":Jz, "hz":mu, "L":L, "shape":SHAPE, "adiabatic_time":ADIABATIC_TIME, "boundary_potential":bp_h, "Jxx_coeff":J,"Jz_coeff":Jz})
    #M_i = xxzhs.AdiabaticHamiltonian({"Jxx":J, "Jz":Jz, "hz":mu, "L":L, "shape":SHAPE, "adiabatic_time":ADIABATIC_TIME,"Jxx_coeff":J,"Jz_coeff":Jz})
    M_f = xxzhs.XXZChainWithCenterPotential({'Jxx':J,"Jz":Jz,'hz':mu,"L":L,"boundary_potential":bp_h})

    #run_data, initial_state = aft.complete_adiabatic_evolution_run(M_i, M_f, initial_guess_lat, dmrg_params, tebd_params, ADIABATIC_TIME, verbose=False, initial_state=initial_state_guess)
    run_data, initial_state = aft.complete_adiabatic_evolution_run(M_i, M_f, initial_guess_lat, dmrg_params, tebd_params, ADIABATIC_TIME, verbose=True)

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
                RENAME_DATA,RENAME_DATA_2 = rt.rodeo_run(rodeo_H, initial_state, goal_state, E_target, time_samples, tebd_params)
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
            print(1/calculated_sp*(ADIABATIC_TIME + r*sigma))
        #print("standard deviation")
        #iiiii = 0
        #while True:
        #    try:
        #        print(np.std(np.array(y_vals_2)[:,iiiii]))
        #        iiiii+=1
        #    except:
        #        break
        #plt.plot(E_target_vals,np.mean(y_vals,axis=0),label="overlap")
        #for yvalset in y_vals:
        #   plt.scatter(E_target_vals,yvalset,label="overlap")
        #plt.plot(E_target_vals,np.mean(y_vals_2,axis=0),label="success_chance")
        #for yvalset2 in y_vals_2:
        #    plt.scatter(E_target_vals,yvalset2,label="success_chance")
        #plt.legend()
        #plt.show()

        TOWRITENAME = "adiabatic_rodeo_results.dat"
        with open(TOWRITENAME,"a") as f:
            towritestring = ""
            for shapev in SHAPE:
                towritestring += str(shapev)
            towritestring += ','
            towritestring += str(sigma) + "," + str(r) + "," + str(ADIABATIC_TIME) + "," + str(resamples) + "," + OVERLAP_TO_WRITE + "," + SUCCESS_CHANCE_TO_WRITE + "," + str(tebd_params['dt']) + "\n"
            f.write(towritestring)


    #plt.plot(r_vals, y_vals_FAKE)
    #plt.show()

def from_argv():
    import sys
    sys.argv[1]
    

if __name__ == "__main__":
    # Model parameters
    J = 1
    SHAPE = [2]*2
    #SHAPE = [2]*2
    mu = 0
    Jz = 0 #-2 #-3

    # Simulation parameters
    #E_target_vals = np.linspace(-1.120,-1.100,8)
    #E_target_vals = np.linspace(-1.15,-1.05,10)
    E_target_vals = np.linspace(-1.1180,-1.1179,1)
    #E_target_vals = np.linspace(-0.8403970657345665, -0.8403970657345665, 1)
    #E_target_vals = np.linspace(-2.379385241571817, -2.379385241571817, 1)
    #E_target_vals = np.linspace(-4.9189757237297185, -4.9189757237297185, 1)
    #E_target_vals = np.linspace(-10.008193950241653,-10.008193950241653,1)
    #E_target_vals = np.linspace(-0.6861406616345073,-0.6861406616345073,1)
    #E_target_vals = np.linspace(-10.008193,-10.008193,1)
    #E_target_vals = np.linspace(-4.91898,-4.91898,2)
    #E_target_vals = np.linspace(-4.95,-4.9,5)
    #E_target_vals = np.linspace(-1.5,-0.5,20)

    sigma = 1
    r_vals = [6]
    resamples = 10
    ADIABATIC_TIME = 0

    run(J, Jz, SHAPE, mu, E_target_vals, sigma, r_vals, resamples, ADIABATIC_TIME)


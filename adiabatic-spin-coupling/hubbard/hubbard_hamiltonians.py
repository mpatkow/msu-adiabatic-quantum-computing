import numpy as np
from tenpy.models.xxz_chain import XXZChain2 as XXZChain
from tenpy.models.hubbard import FermiHubbardChain

def get_non_interaction_term_indicies(initial_state):
    indicies = []

    i = -1
    for block_length in initial_state:
        i+=block_length
        indicies.append(i)

    return indicies[:-1]

# An XXZChain but interaction terms (both Jxx and Jz)
# are progressively turned on throughout the adiabatic evolution
class AdiabaticHubbardHamiltonian(FermiHubbardChain):
    def init_terms(self, model_params):
        c_arr = np.ones(model_params['L']-1)

        for non_coupling_index in get_non_interaction_term_indicies(model_params['shape']):
            if model_params['adiabatic_time'] != 0:
                c_arr[non_coupling_index] = model_params.get('time', 0)/model_params['adiabatic_time']
            else:
                c_arr[non_coupling_index] = 0

        model_params['t'] = c_arr * model_params['t_coeff']
        #model_params['U'] = c_arr * model_params['U'] 
        #model_params['mu'] = c_arr * model_params['mu'] 
        #model_params['V'] = c_arr * model_params['V'] 

        super().init_terms(model_params)

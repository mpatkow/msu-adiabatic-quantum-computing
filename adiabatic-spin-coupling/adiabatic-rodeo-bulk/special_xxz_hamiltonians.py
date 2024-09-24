import numpy as np
from tenpy.models.xxz_chain import XXZChain2 as XXZChain

def get_non_interaction_term_indicies(initial_state):
    indicies = []

    i = -1
    for block_length in initial_state:
        i+=block_length
        indicies.append(i)

    return indicies[:-1]

# An XXZChain but interaction terms (both Jxx and Jz)
# are progressively turned on throughout the adiabatic evolution
class AdiabaticHamiltonian(XXZChain):
    def init_terms(self, model_params):
        c_arr = np.ones(model_params['L']-1)

        for non_coupling_index in get_non_interaction_term_indicies(model_params['shape']):
            if model_params['adiabatic_time'] != 0:
                c_arr[non_coupling_index] = model_params.get('time', 0)/model_params['adiabatic_time']
            else:
                c_arr[non_coupling_index] = 0

        model_params['Jxx'] = c_arr * model_params['Jxx_coeff']
        model_params['Jz'] = c_arr * model_params['Jz_coeff'] 

        super().init_terms(model_params)

class AdiabaticHamiltonianWithSlantPotential(AdiabaticHamiltonian):
    def init_terms(self, model_params):
        h_values = np.asarray([self.interp_func_for_h_slant(site_i, model_params["boundary_potential"], model_params["L"]) for site_i in range(model_params["L"])])
        print(h_values)
        self.add_onsite(h_values, 0, 'Sz')

        super().init_terms(model_params)

    def interp_func_for_h_abs(self, i, h, L):
        return h*np.abs(i-L/2+1/2)

    def interp_func_for_h_slant(self, i, h, L):
        return h*i

class XXZChainWithSlantPotential(XXZChain):
    def init_terms(self, model_params):
        h_values = np.asarray([self.interp_func_for_h_slant(site_i, model_params["boundary_potential"], model_params["L"]) for site_i in range(model_params["L"])])
        print(h_values)
        self.add_onsite(h_values, 0, 'Sz')

        super().init_terms(model_params)

    def interp_func_for_h_abs(self, i, h, L):
        return h*np.abs(i-L/2+1/2)

    def interp_func_for_h_slant(self, i, h, L):
        return h*i

import json
import os
import adiabatic_evolution_tenpy_new
from tenpy.models.tf_ising import TFIChain
from tenpy.models.xxz_chain import XXZChain2
import numpy as np
from tqdm import tqdm

FILENAME = "input_parameters_2.json"
OUTPUTDIR = "results-2"
f = open(FILENAME,)
all_param_set = json.load(f)

# Model for the transverse ising chain that linearly interpolates the coupling terms based on shape provided
class AdiabaticHamiltonian(XXZChain2):
    def init_terms(self, model_params):
        c_arr = np.ones(L-1)
        for non_coupling_index in adiabatic_evolution_tenpy_new.get_non_interaction_term_indicies(shape):
            if T != 0:
                c_arr[non_coupling_index] = model_params.get('time', 0)/T
            else:
                c_arr[non_coupling_index] = 0

        model_params['Jxx'] = c_arr * J
        print("!!!")
        print(model_params['Jxx'])

        super().init_terms(model_params)

for key in tqdm(all_param_set.keys()):
    single_shot_params = all_param_set[key]

    T = single_shot_params['T']
    J = single_shot_params['J']
    h = single_shot_params['h']
    shape = single_shot_params['shape']
    dmrg_p = single_shot_params['dmrg_param']
    tebd_p = single_shot_params['tebd_param']

    shape_encoded = ""
    for s in shape:
        shape_encoded += str(s)
    file_name = key + "_" + "T" + str(T) + "_" + "J" + str(J) + "_" + "h" + str(h) + "_" + "shape" + shape_encoded
    file_name = file_name.replace(".","p")
    file_name = file_name.replace("-","m")

    if os.path.isfile(OUTPUTDIR+"/"+file_name+".txt"):
        continue

    L = sum(shape)
   
    M_i = AdiabaticHamiltonian({'Jxx':J, 'Jz':0,'hz':h,"L":L})
    M_f = XXZChain2({'Jxx':J,'Jz':0, 'hz':h,"L":L})
    run_data = adiabatic_evolution_tenpy_new.complete_adiabatic_evolution_run(M_i, M_f, dmrg_p, tebd_p, T,verbose=False)

    with open(OUTPUTDIR+"/" + file_name+'.txt','w') as data_file:  
        data_file.write(str(run_data))

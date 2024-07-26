import json
import numpy as np

h_values = [0.5]
total_runtime_values = np.linspace(1,30,120)
j_values = [-1]
shape_values = [[2]*2, [2]*4, [2]*8, [2]*16, [2]*32, [4]*2, [4]*4, [4]*8, [4]*16, [8]*2,[8]*4,[8]*8,[16]*2,[16]*4,[32]*2]

dmrg_default_params = {
    'mixer': None,  # setting this to True helps to escape local minima
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 100,
        'svd_min': 1.e-10,
    },
    'combine': True,
}

tebd_default_params = {
    'N_steps': 1,
    'dt': 1,
    'order': 4,
    'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
}

dmrg_params = [dmrg_default_params]
tebd_params = [tebd_default_params]

simulation_params = {}
i=0
for shape_value in shape_values:
    for total_runtime_value in total_runtime_values:
        for j_value in j_values:
            for h_value in h_values:
                for dmrg_param in dmrg_params:
                    for tebd_param in tebd_params:
                        full_param = {}
                        full_param['T'] = total_runtime_value
                        full_param['h'] = h_value
                        full_param['J'] = j_value
                        full_param['shape'] = shape_value
                        full_param['dmrg_param'] = dmrg_param
                        full_param['tebd_param'] = tebd_param

                        simulation_params[i] = full_param

                        i+=1

with open("input_parameters.json", "a") as outfile:
    json.dump(simulation_params, outfile)



import json
import os
import numpy as np

data = {
'initial_guess_lat': [['down'],['up']], #[['down'],['down'],['down'],['up']],#[['down'],['up']],
'dmrg_params' : {
'mixer': None,  # setting this to True helps to escape local minima
'max_E_err': 1.e-10,
'trunc_params': {
    'chi_max': 100,
    'svd_min': 1.e-10,
},
'combine': True,
},
'special_xxzmodel_params' : {
    'shape' : [128]*2,
    'Jxx_coeff' : 1,
    'Jxx' : 1,
    'Jz_coeff' : 0,
    "Jz" : 0,
    'hz' : 0,
    'boundary_potential' : 0,
},
'adiabatic_params' : {
'adiabatic_time' : 7,
'verbose' : True,
'tebd_params' : {
    'N_steps': 1,
    'dt': 0.1,
    'order': 4,
    'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
    },
},
'rodeo_params' : {
'resamples' : 1,
'r_value' : 0,
'r_superiteration' : 6,
'sigma' : 3, #5, #41, #10.5, #5.42, # 2.86, #1.61 
'e_target' : #-4.9189757237297185, #-1.1180339887498945, #-0.9102295587243708, #-2.833270928996819, #-10.0081939502417, #-7.105137640800324, #-1.7057370639048854, #,-0.8090169943749473, #-20.192156579895645, #-40.56298999660997, #-10.008193950241653, #-4.9189757237297185,
'tebd_params' : {
        'N_steps': 1,
        'dt': 0.1,
        'order': 4,
        'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
    },
},
"verbose" : True,
"write_to_outfile" : False,
"outfile" : "adiabatic_rodeo_results.csv",
"saved_mps_bool" : True,
"save_mps_bool" : False,
"saved_mps_filename" : "asdf.pkl",
"super_iterations" : True,
}

data['special_xxzmodel_params']['L'] = sum(data['special_xxzmodel_params']['shape'])

with open(f'adiabatic_rodeo_params.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


import json
import os
import numpy as np

#adiabatic_times = np.concatenate((np.linspace(0,30,31), np.linspace(30,50,11)))
adiabatic_times = np.linspace(120,120,1) #[0]
#r_values = [int(j) for j in np.linspace(1,1,1)]
r_values = [0,1,2]
sigma_values = [2,4,6] #[17.984] #[164] # using 3.3 for the [2,2] case using 5.76 for the [4,4] case. 10.8 for the [8,8] case, 21 for [16,16] case, 82 for [64,64], 164 for [128,128]
i = 0
for adiabatic_time in adiabatic_times:
    for r_value in r_values:
        for sigma_value in sigma_values:
            data = {
            'initial_guess_lat': [['up'],['empty'],['down'],['empty']], #[['down'],['down'],['down'],['up']],#[['down'],['up']],
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
                'shape' : [32]*2,
                'cons_N' : 'N',
                'cons_Sz' : 'Sz',
                'mu' : 0,
                'V' : 0,
                't' : 1,
                'U': -1,
                't_coeff' : 1,
            },
            'adiabatic_params' : {
            'adiabatic_time' : adiabatic_time,
            'verbose' : True,
            'tebd_params' : {
                'N_steps': 1,
                'dt': 0.1, # VERY LOW ACCURACY!!!!!
                'order': 4,
                'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
                },
            },
            'rodeo_params' : {
            'resamples' : 1,
            'r_value' : r_value,
            'r_superiteration' : 20,
            'sigma' : sigma_value, #5, #41, #10.5, #5.42, # 2.86, #1.61 
            'e_target' : -81.30615008878445, #-2.067454681324168, #-0.9484472909001022, #-2.1774096323578274, #-1.9995638775314657, #-0.9098901883856574, #-1.7057370639048854, #-0.8090169943749473, #-10.0081939502417, #-4.918975723729717, #-2.379385241571817, #-1.1180339887498945, #-81.30614838631675, #-40.56298999660997, #-20.192156579894892, #-10.0081939502417, #-2.379385241571817, #-1.1180339887498945, #-1.1180339887498945,#-10.008193950241653, #-4.918975723729717, #-2.379385241571817, #-1.1180339887498945, #-0.9102295587243708, #-2.833270928996819, #-10.0081939502417, #-7.105137640800324, #-1.7057370639048854, #,-0.8090169943749473, #-20.192156579895645, #-40.56298999660997, #-10.008193950241653, #-4.9189757237297185,
            'tebd_params' : {
                    'N_steps': 1,
                    'dt': 0.1, # VERY LOW ACCURACY!!!
                    'order': 4,
                    'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
                },
            },
            "verbose" : True,
            "write_to_outfile" : False,
            "outfile" : "adiabatic_rodeo_results.csv",
            "saved_mps_bool" : False,
            "save_mps_bool" : False,
            "saved_mps_filename" : "asdfQUARTER.pkl",
            "super_iterations" : True,
            "use_dmrg_etarget" : True,
            }

            data['special_xxzmodel_params']['L'] = sum(data['special_xxzmodel_params']['shape'])

            while os.path.isfile(f"data/adiabatic_rodeo_params_{i}.json"):
                i+=1
            with open(f'data/adiabatic_rodeo_params_{i}.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

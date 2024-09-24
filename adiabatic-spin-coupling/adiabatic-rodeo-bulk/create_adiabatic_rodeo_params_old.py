import json
import os
import numpy as np

#adiabatic_times = np.linspace(6,6,1)
adiabatic_times = np.linspace(15,25,11)
r_values = [int(j) for j in np.linspace(1,10,10)]
sigma_values = [3]
i = 0
for adiabatic_time in adiabatic_times:
    for r_value in r_values:
        for sigma_value in sigma_values:
            data = {
            'initial_guess_lat': [['down'],['up']],
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
                    'shape' : [2]*2,
                    'Jxx_coeff' : 1,
                    'Jxx' : 1,
                    'Jz_coeff' : 0,
                    "Jz" : 0,
                    'hz' : 0,
                    'boundary_potential' : 0,
                },
            'adiabatic_params' : {
                'adiabatic_time' : adiabatic_time,
                'verbose' : True,
                'tebd_params' : {
                    'N_steps': 1,
                    'dt': 0.1,
                    'order': 4,
                    'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
                    },
                },
            'rodeo_params' : {
                'resamples' : 0,
                'r_value' : r_value,
                'sigma' : sigma_value,
                'e_target' : -2.3793852,
                'tebd_params' : {
                        'N_steps': 1,
                        'dt': 0.1,
                        'order': 4,
                        'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
                    },
                },
            "verbose" : True,
            "write_to_outfile" : False,
            "outfile" : "adiabatic_rodeo_results.csv"
            }

            data['special_xxzmodel_params']['L'] = sum(data['special_xxzmodel_params']['shape'])

            while os.path.isfile(f"data/adiabatic_rodeo_params_{i}.json"):
                i+=1
            with open(f'data/adiabatic_rodeo_params_{i}.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


import matplotlib.pyplot as plt
import numpy as np

"""
results generated with general params of : (with varying sizes, sigmas, etc.)
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
    'shape' : [4]*2,
    'Jxx_coeff' : 1,
    'Jxx' : 1,
    'Jz_coeff' : 0,
    "Jz" : 0,
    'hz' : 0,
    'boundary_potential' : 0,
},
'adiabatic_params' : {
'adiabatic_time' : 10,
'verbose' : True,
'tebd_params' : {
    'N_steps': 1,
    'dt': 0.1,
    'order': 4,
    'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
    },
},
'rodeo_params' : {
'resamples' : 100,
'r_value' : 3,
'sigma' : 2.86, #1.61 
'e_target' : -2.379385241571817,
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
"saved_mps_filename" : "asdf.pkl"
}
"""

#Ls = [4,8,16,32]
#adiabatic_times = [6,10,18,34]
#e4 = [10.867015394922651, 24.818313017494884, 43.03799649327762, 85.65905152549655]
Ls = [8,16,32]
adiabatic_times_e4 = [7, 13, 51] # datapoint for Ls=32 is most likely not optimal
m,b = np.polyfit(np.log2(Ls), np.log10(adiabatic_times_e4), 1)
print([10**(m*Lval + b) for Lval in [3,4,5,6]])
print(adiabatic_times_e4)
#adiabatic_times_e3 = [6, 18.46]
adiabatic_times_e2 = [7, 12, 24]
e4 = [12.44, 27.71, 85] # datapoint for Ls=32 is most likely not optimal
#e3 = [6, 18.46]
e2 = [7, 12, 24] 
e4_rudimentary_full = [sum(e4[:i+1]) for i in range(len(e4))]
#e3_rudimentary_full = [sum(e3[:i+1]) for i in range(len(e3))]
e2_rudimentary_full = [sum(e2[:i+1]) for i in range(len(e2))]
#print(e4_rudimentary_full)
#resamples = [100,10,10,"fake"] # used to confirm accuracy of data

plt.plot(np.log2(Ls), e4, label="total_times_e4")
#plt.plot(np.log2(Ls), e4, label="total_times_e3",ls="dashed")
plt.plot(np.log2(Ls), e4, label="total_times_e2",linestyle="dotted")
plt.plot(np.log2(Ls), adiabatic_times_e4, label="adiabatic_times")
#plt.plot(np.log2(Ls), adiabatic_times_e3, label="adiabatic_times",ls="dashed")
plt.plot(np.log2(Ls), adiabatic_times_e2, label="adiabatic_times",ls="dotted")
plt.plot(np.log2(Ls), e4_rudimentary_full, label="total_times_e4_rudimentary_full")
#plt.plot(np.log2(Ls), e3_rudimentary_full, label="total_times_e3_rudimentary_full",ls="dashed")
plt.plot(np.log2(Ls), e2_rudimentary_full, label="total_times_e2_rudimentary_full",ls="dotted")
plt.xlabel(r"$\log_2 L$")
plt.ylabel(r"$T$")
plt.yscale('log')
plt.legend()
plt.show()

import adiabatic_rodeo
import threading
import numpy as np


J = 1
Jz = -0.5
SHAPE = [2]*2
mu = 0
#E_target_vals = np.linspace(-1.1180,-1.1179,1)
E_target_vals = np.linspace(-0.6861,-0.6861,1)
#E_target_vals = np.linspace(-0.75,-0.5, 5)
#E_target_vals = np.linspace(-1.2,-1.0,5)
#E_target_vals = np.linspace(-11, -9, 10)
sigma_vals = [1]#np.linspace(0,4,40) #,5,10,20]
r_vals_vals = [[0],[1],[2],[3],[4],[5]] # ,[4],[5],[6],[7]]
resamples = 5
adiabatic_times = np.linspace(0,10,10)
#adiabatic_times = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5]
#adiabatic_times = [6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,13,14,15]

for r_vals in r_vals_vals:
    for ADIABATIC_TIME in adiabatic_times:
        for sigma in sigma_vals:
            print(f"sigma: {sigma}, rvals: {r_vals}, adiabatic_time {ADIABATIC_TIME}")
            adiabatic_rodeo.run(J,Jz, SHAPE, mu, E_target_vals, sigma, r_vals, resamples, ADIABATIC_TIME)
            #t = threading.Thread(name='child procs', target=adiabatic_rodeo.run, args = (J, SHAPE, mu, E_target_vals, sigma, r_vals, resamples, ADIABATIC_TIME))
            #t.start()
        




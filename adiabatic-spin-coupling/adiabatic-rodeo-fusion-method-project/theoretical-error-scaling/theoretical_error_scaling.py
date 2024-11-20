import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fsolve as fsolve

def error_AE(t_ae, error_0):
    p_eq = lambda p : 1 - p - error_0/(t_ae*p + 1)**2

    p_sol = fsolve(p_eq, 0.9)

    return 1-p_sol

def error_AR(t_ae, t_ra, sigma, error_0):
    e_AA = error_AE(t_ae, error_0)
    p = 1-e_AA

    error_val = 1/2**(t_ra * p / sigma) * e_AA

    return error_val

def main():
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    matplotlib.rc('font', **font)

    error_0 = 0.5
    sigma = 4

    # L = 4
    #error_0 = 0.10279
    #sigma = 3.3/2
    
    # L = 8
    #error_0 = 0.15848
    #sigma = 5.76/2

    # L = 16
    #error_0 = 0.217
    #sigma = 10.8/2

    # L = 32
    #error_0 = 0.27531521
    #sigma = 21/2

    # L = 64 
    #error_0 = 0.331444388
    #sigma = 42/2

    # L = 128
    #error_0 = 0.3846
    #sigma = 84/2

    # L = 256
    error_0 = 1-0.5656140020714872
    sigma = 168/2


    #error_0 = 0.8
    #sigma = 1

    error_t_s = [10**(-i) for i in np.linspace(0,10,50)]
    #error_t_s = [10**(-i) for i in np.linspace(0,4,5)]
    t_min_total_s = []

    for i,error_t in enumerate(error_t_s):
        T_AE_s = np.linspace(0,100,300)
        T_RA_sol_s = []

        for T_AE in T_AE_s:
            error_root_func = lambda t : error_AR(T_AE, t, sigma, error_0) - error_t

            T_RA_sol = fsolve(error_root_func, 0)

            T_RA_sol = float(max(T_RA_sol, 0))
            
            T_RA_sol_s.append(T_RA_sol)
            #error_AE_vals.append(error_AR(T_AE, T_RA, sigma, error_0))

        T_total_s = T_RA_sol_s + T_AE_s
        plt.plot(T_AE_s, T_total_s, label=str(error_t))
        min_i = np.argmin(T_total_s)

        plt.plot(T_AE_s[min_i], T_total_s[min_i], "ok", linewidth=5)

        t_min_total_s.append(T_total_s[min_i])
        
        print(f"Error: {error_t} reached in total {T_total_s[min_i]} time with {T_AE_s[min_i]/T_total_s[min_i]}% time in adiabatic")

    plt.xlabel("Adiabatic Time")
    plt.ylabel("Total Time")
    plt.legend()
    plt.show()


    plt.xlabel("Total Time")
    plt.ylabel("Target Error")
    plt.semilogy(t_min_total_s, error_t_s)
    plt.show()
    
if __name__ == "__main__":
    main()

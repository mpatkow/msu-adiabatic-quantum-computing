import numpy as np
import matplotlib.pyplot as plt
import math

a_set = [1-0.897213595499958,1-0.841519445765513,1-0.7828712176562818,1-0.724684781177863,1-0.6685556122386582] # initial error
#T_C_set = [3.3,5.76,10.8,21,42]
T_C_set = [6.6,2*5.76,2*10.8,2*21,2*42]

for T_C,a in zip(T_C_set,a_set):
    eps_F_set = []
    T_T_set = []
    T_A_set = []
    #eps_A_set = np.linspace(0,1,1000)
    eps_A_set = np.linspace(1e-4,a,100000)
    for eps_A in eps_A_set:
        alpha = 0.05
        #eps_A = 0.01
        #alpha = 0.5
        eps_A_star = 0.01 

        #p = 1.33
        #p = 1.2169
        #p = 1.406298
        p = 0.77978
        eps_F = 1e-4
        try:
            #N = math.ceil(np.log(eps_F/eps_A)/np.log(alpha))
            N = np.log(eps_F/eps_A)/np.log(alpha)
        except:
            N=0
        #T_C = 2
        #b = 0.304
        #b = .2409
        #b = 0.15942
        b = 0.103316

        if eps_A > eps_A_star:
            T_A = np.log(eps_A / a) / (-b)
        else:
            T_A = np.log(eps_A_star / a) / (-b)
            c = eps_A_star * T_A**p
            T_A = (c / (eps_A)) ** (1/p)
        if N == 0:
            T_T = T_A
        else:
            T_T = 1/(1-eps_A) * (T_A + N*T_C)

        #eps_F = eps_A * alpha ** N
        eps_F_set.append(eps_F)
        T_T_set.append(T_T)
        T_A_set.append(T_A)

    #plt.plot(eps_A_set, eps_F_set)
    #plt.show()

    plt.plot(eps_A_set, T_T_set)
    print(f"Minimum time: {min(T_T_set)}")
    print(f"Minimum adiabatic time {T_A_set[np.argmin(T_T_set)]}")
    plt.xscale("log")
    plt.show()

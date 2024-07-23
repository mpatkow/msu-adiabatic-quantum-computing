import numpy as np
import tenpy

L = 4 # or whatever you like...
model_params = {
    'L': L,
    'Jx': 1., 'Jy': 0., 'Jz': 0.,
    'hz': np.ones(L),
    'conserve': 'best',
}
M = tenpy.models.spins.SpinChain(model_params)


def charge_density_wave(L):
    """Initial state |010101010...>"""
    psi0 = np.mod(np.arange(0,L),2)
    return np.diag(psi0)

print(M)
sim = tenpy.simulations.time_evolution.RealTimeEvolution({"final_time":1})
sim.run_algorithm()

#print(tenpy.algorithms.exact_diag.ExactDiag(M))
#asdf = tenpy.algorithms.mpo_evolution.ExpMPOEvolution(charge_density_wave(L),M, {})
#print(asdf.calc_U(1))

#asdf.run_evolution(1,1)

import numpy as np
import scipy
import matplotlib.pyplot as plt
#np.set_printoptions(precision=5, suppress=True, linewidth=100)
#plt.rcParams['figure.dpi'] = 150

import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import tebd

tenpy.tools.misc.setup_logging(to_stdout="INFO")

L = 4
h = 1
TOTAL_TIME = 5

model_params = {
    'J': 0.,
    'g': h,
    'L': L,
    'bc_MPS': 'finite',
    #'bc_x' : 'periodic'
}

model_params_2 = {
    'J': -1.,
    'g': h,
    'L': L,
    'bc_MPS': 'finite',
    #'bc_x' : 'periodic'
}

class AdiabaticHamiltonian(TFIChain):
    def init_terms(self, model_params):
        
        J = -(model_params.get("time", 0.)/TOTAL_TIME)
        print(J)

        self.add_coupling_term(-1, 0, 1, "Sx", "Sx")
        self.add_coupling_term(-1, 2, 3, "Sx", "Sx")
        self.add_coupling_term(J, 1, 2, "Sx", "Sx")
        #j..print(self.lat.pairs["nearest_neighbors"])
        #for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
        #    self.add_coupling(J, u1, 'Sx', u2, 'Sx', dx)

        #super().init_terms(model_params)


M = TFIChain(model_params)
#M = AdiabaticHamiltonian(model_params)
#M.init_terms(model_params)
M_full = TFIChain(model_params_2)

#M.manually_call_init_H = True 
M.add_coupling_term(-1, 0, 1, "Sx", "Sx")
M.add_coupling_term(-1, 2, 3, "Sx", "Sx")
M.add_coupling_term(-1, 1, 2, "Sx", "Sx")
M.init_H_from_terms()

psi_guess = MPS.from_lat_product_state(M.lat, [['up']])

dmrg_params = {
    'mixer': None,  # setting this to True helps to escape local minima
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 100,
        'svd_min': 1.e-10,
    },
    'combine': True,
}
eng1 = dmrg.TwoSiteDMRGEngine(psi_guess, M, dmrg_params)
E, psi_start = eng1.run() # the main work; modifies psi in place

eng2 = dmrg.TwoSiteDMRGEngine(psi_start.copy(), M_full, dmrg_params)
E2, psi_actual = eng2.run() # the main work; modifies psi in place

# the ground state energy was directly returned by dmrg.run()
print("ground state energy = ", E)
print("ground state energy = ", E2)

tebd_params = {
    'N_steps': 1,
    'dt': 0.1,
    'order': 4,
    'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
}

#eng = tebd.TEBDEngine(psi_start, M, tebd_params) # TODO should the engine be updated every time?
eng = tebd.TimeDependentTEBD(psi_start, M, tebd_params) # TODO should the engine be updated every time?

def measurement(eng, data):
    keys = ['t', 'entropy', 'Sx', 'Sz', 'corr_XX', 'corr_ZZ', 'trunc_err', 'overlap']
    if data is None:
        data = dict([(k, []) for k in keys])
    data['t'].append(eng.evolved_time)
    data['entropy'].append(eng.psi.entanglement_entropy())
    data['Sx'].append(eng.psi.expectation_value('Sigmax'))
    data['Sz'].append(eng.psi.expectation_value('Sigmaz'))
    data['corr_XX'].append(eng.psi.correlation_function('Sigmax', 'Sigmax'))
    data['trunc_err'].append(eng.trunc_err.eps)
    overlap_unsq = eng.psi.overlap(psi_actual)
    print(f"Unsquared Overlap: {overlap_unsq}")
    print(f"Overlap: {overlap_unsq.conj()*overlap_unsq}")
    data['overlap'].append(overlap_unsq.conj()*overlap_unsq)
    #print(eng.psi.get_B(0))
    return data

data = measurement(eng, None)


while eng.evolved_time < TOTAL_TIME:
    eng.run()
    
    print(measurement(eng, data))
    #M.add_coupling_term(-tebd_params['dt']/TOTAL_TIME, 1, 2, "Sx", "Sx")
    #M.init_terms(model_params)
    #M.update_time_parameter(eng.evolved_time)
    #M.init_H_from_terms()

plt.plot(data['t'], data['overlap'])
plt.ylim(-0.1, 1.1)
plt.show()

# Attempt to create and adiabatically evolve Ising chains in tenpy

import numpy as np
import scipy
import matplotlib.pyplot as plt
import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain, TFIModel
from tenpy.algorithms import tebd
from tenpy.models.model import CouplingMPOModel
from tenpy.models.spins import SpinModel, SpinChain
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel

tenpy.tools.misc.setup_logging(to_stdout="INFO")




L = 4
h = 1
TOTAL_TIME = 5
J = -1
c_arr = np.ones(L-1,dtype=float)
c_arr[1]= 0

model_params = {
    'J': J*c_arr,
    'g': h,
    'L': L,
    #'bc_MPS': 'periodic',
    #'bc_x' : 'periodic'
}

model_params_spin_chain = {
    'bc_MPS': 'finite',
    'Jx': -J,
    'Jy': 0,
    'Jz': 0,
    'hx': 0,
    'hy': 0,
    'hz': h,
    'L': L,
    'S': 1/2,
    'muJ': 0,
    "D": 0,
    "E": 0,
}

model_params_2 = {
    'J': -1.,
    'g': h,
    'L': L,
    'bc_MPS': 'finite',
    #'bc_x' : 'periodic'
}

class MyTimeDepModel(TFIChain):
    def init_terms(self, model_params):
        c_arr = np.ones(L-1)
        c_arr[1] = model_params.get('time', 0)/TOTAL_TIME

        model_params['J'] = c_arr * -1 # TODO CHANGE TO UPDATE TO J Value

        #print(self.lat.pairs)
        #print([type(v) for v in self.lat.pairs['nearest_neighbors'][0]])
        #self.init_H_from_terms()
        super().init_terms(model_params)

class MyTimeDepModel2(TFIChain):
    def init_terms(self, model_params):
        #J = -(model_params.get("time", 0.)/TOTAL_TIME)
        #print(J)

        #print(self.lat.pairs)
        #for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
        #    self.add_coupling(-1, u1, 'Sx', u2, 'Sx', dx)
        #self.add_coupling_term(-1, 0, 1, "Sx", "Sx")
        #self.add_coupling_term(-1, 2, 3, "Sx", "Sx")
        #self.add_coupling_term(J, 1, 2, "Sx", "Sx")
        #j..print(self.lat.pairs["nearest_neighbors"])
        #for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
        #    self.add_coupling(J, u1, 'Sx', u2, 'Sx', dx)

        super().init_terms(model_params)

class XXZChain(CouplingModel, NearestNeighborModel, MPOModel):
    def __init__(self, L=2, S=0.5, J=1, hz=1):
        spin = SpinSite(S=S, conserve="Sz")
        # the lattice defines the geometry
        lattice = Chain(L, spin, bc="open", bc_MPS="finite")
        CouplingModel.__init__(self, lattice)
        # add terms of the Hamiltonian
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-1, u1, 'Sx', u2, 'Sx', dx)
        #self.add_coupling(J, 0, "Sx", 0, "Sx", 1) # Sp_i Sm_{i+1}
        #self.add_coupling(J, 0, "Sp", 0, "Sm", -1) # Sp_i Sm_{i-1}
        #self.add_coupling(J, 0, "Sz", 0, "Sz", 1)
        # (for site dependent prefactors, the strength can be an array)
        self.add_onsite(-hz, 0, "Sz")
        # finish initialization
        # generate MPO for DMRG
        MPOModel.__init__(self, lattice, self.calc_H_MPO())
        # generate H_bond for TEBD
        NearestNeighborModel.__init__(self, lattice, self.calc_H_bond())


model_params_3 = {
    'L': L,
    'Jx': -1*np.ones(L-1), 'Jy': 0., 'Jz': 0.,
    'hz': -1,  # random values in [-W, W], shape (L,)
    'conserve': 'best',
}
M = tenpy.models.spins.SpinChain(model_params_3)


#M = AdiabaticHamiltonian(model_params)
#M = SpinChain(model_params)
M = MyTimeDepModel(model_params)
#M = XXZChain(L=4)
#M = MyTimeDepModel2(model_params)
print(f"Pairs of {M}: {M.lat.pairs}")
#M = CouplingMPOModel(model_params)
#M = AdiabaticHamiltonian(model_params)
#M.init_terms(model_params)
M_full = TFIChain(model_params_2)

#M.manually_call_init_H = True 
#for u1, u2, dx in M.lat.pairs['nearest_neighbors']:
    #print(u1)
#    M.add_coupling(-1, u1, 'Sx', u2, 'Sx', dx)
#M.init_H_from_terms()

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
print(f"Energy of Initial Hamiltonian: {E}")
input()

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
    keys = ['t', 'entropy', 'Sx', 'Sz', 'corr_XX', 'corr_ZZ', 'trunc_err', 'overlap', 'ene_exp']
    if data is None:
        data = dict([(k, []) for k in keys])
    data['t'].append(eng.evolved_time)
    data['entropy'].append(eng.psi.entanglement_entropy())
    #data['Sx'].append(eng.psi.expectation_value('Sigmax'))
    #data['Sz'].append(eng.psi.expectation_value('Sigmaz'))
    #data['corr_XX'].append(eng.psi.correlation_function('Sigmax', 'Sigmax'))
    #data['trunc_err'].append(eng.trunc_err.eps)
    data['ene_exp'].append(M.H_MPO.expectation_value(eng.psi))
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
    M.init_H_from_terms()
    M.update_time_parameter(eng.evolved_time)

plt.subplot(2,1,1)
plt.plot(data['t'], data['overlap'])
plt.ylim(-0.1, 1.1)
plt.subplot(2,1,2)
plt.plot(data['t'], data['ene_exp'])
plt.show()

import numpy as np
import scipy
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, suppress=True, linewidth=120)
import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import tebd
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain, TFIModel
tenpy.algorithms.tebd.TimeDependentTEBD
tenpy.tools.misc.setup_logging(to_stdout="DEBUG")
import yaml

class AdiabaticHamiltonian(TFIModel):
    def init_terms(self, model_params):
        model_params['J'] = 1
        
        J = model_params.get("time", 0.) - 1

        print(self.lat.pairs["nearest_neighbors"])
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J, u1, 'Sx', u2, 'Sx', dx)

        super().init_terms(model_params)

#   bc_x : periodic
#   bc_MPS : periodic
#algorithm_class: TimeDependentExpMPOEvolution

params = """
model_class :  AdiabaticHamiltonian
model_params :
    L : 4
    g : 1
    bc_x : periodic
    explicit_plus_hc : False

initial_state_params:
    method : lat_product_state
    product_state : [[down]]

algorithm_class : TimeDependent
algorithm_params:
    trunc_params:
        chi_max: 120
    dt : 0.05
    N_steps : 2
    compression_method: variational

connect_measurements:
    - - tenpy.simulations.measurement
      - m_energy_MPO

final_time :  1
"""
sim_params = yaml.safe_load(params)

res = tenpy.run_simulation('RealTimeEvolution', **sim_params)
print(res)
print(res['psi'])


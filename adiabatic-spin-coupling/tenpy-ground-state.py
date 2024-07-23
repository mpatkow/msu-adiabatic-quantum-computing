from tenpy.simulations.ground_state_search import GroundStateSearch
import yaml
from tenpy.models.tf_ising import TFIChain, TFIModel
import tenpy
import numpy as np

from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig

class AdiabaticHamiltonian(TFIModel):
    def init_terms(self, model_params):
        model_params['J'] = -1 
        
        J = -(model_params.get("time", 0.))

        print(self.lat.pairs["nearest_neighbors"])
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J, u1, 'Sx', u2, 'Sx', dx)

        super().init_terms(model_params)


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

algorithm_class : GroundStateSearch
algorithm_params:
    trunc_params:
        chi_max: 120
    

connect_measurements:
    - - tenpy.simulations.measurement
      - m_energy_MPO

final_time :  1
"""


#
#res = tenpy.run_simulation('GroundStateSearch', **sim_params)
#print(res)
#print(res['psi'])


def example_run_dmrg(sim_params):
    """Use iDMRG to extract information about the ground state of the system."""
    model_params = sim_params
    model = AdiabaticHamiltonian(sim_params)
    psi = MPS.from_lat_product_state(model.lat, [["down"]])
    dmrg_params = {
        'mixer': True,
        'chi_list': {
            0: 100
        },
        'trunc_params': {
            'svd_min': 1.e-10
        },
    }
    results = dmrg.run(psi, model, dmrg_params)
    print("Energy per site: ", results['E'])
    print("<Sz>: ", psi.expectation_value('Sz'))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    sim_params = yaml.safe_load(params)
    example_run_dmrg(sim_params)

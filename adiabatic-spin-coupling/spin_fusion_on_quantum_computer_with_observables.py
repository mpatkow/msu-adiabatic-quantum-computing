from qiskit import QuantumCircuit
import sys
from qiskit.primitives import BackendSampler as BackendSampler
from qiskit.primitives import BackendEstimatorV2 as BackendEstimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2, FakeLagosV2
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import matplotlib as mpl
import spin_chain_improved
import trotterization_exact
from qiskit.primitives import StatevectorEstimator as SEstimator
from qiskit.result import sampled_expectation_value

# Perform an arbitrary unitary operation on two qubits
# of the form N = exp(i(alpha sx x sx + beta sy x sy + gamma sz x sz) dt)
def full_N(q_c, bits, J, dt):
    alpha = J[0]
    beta = J[1]
    gamma = J[2]

    alpha *= dt
    beta *= dt
    gamma *= dt
    
    b0 = bits[0]
    b1 = bits[1]


    theta = np.pi/2 - 2*gamma
    phi = 2*alpha - np.pi/2
    lambd = np.pi/2 - 2*beta

    q_c.rz(-np.pi/2, b1)
    q_c.cx(b1, b0)
    q_c.rz(theta, b0)
    q_c.ry(phi, b1)
    q_c.cx(b0, b1)
    q_c.ry(lambd, b1)
    q_c.cx(b1, b0)
    q_c.rz(np.pi/2, b0)

"""
For later use when reducing two two CNOTS 
Consider using the U gate provided by qiskit

def simplified_N(q_c, bits, alpha, gamma):
    b0 = bits[0]
    b1 = bits[1]

    q_c.cx(b0, b1)
    q_c.rx(-2*alpha, b0)
    q_c.rz(-2*gamma, b1)
    q_c.cx(b0, b1)
"""

"""
final_hamiltonian_entries = []
for i in range(SIZE):
    string_of_ops = "I"*i+"Z"+"I"*(SIZE-i-1)
    if CONSTANT_H != 0:
        final_hamiltonian_entries.append((string_of_ops,-CONSTANT_H))

for i in range(SIZE-1):
    string_of_ops_X = "I"*i + "XX" + "I"*(SIZE-i-2)
    string_of_ops_Y = "I"*i + "YY" + "I"*(SIZE-i-2)
    string_of_ops_Z = "I"*i + "ZZ" + "I"*(SIZE-i-2)
    if J[0] != 0:
        final_hamiltonian_entries.append((string_of_ops_X,-J[0]))
    if J[1] != 0:
        final_hamiltonian_entries.append((string_of_ops_Y,-J[1]))
    if J[2] != 0:
        final_hamiltonian_entries.append((string_of_ops_Z,-J[2]))

final_hamiltonian = SparsePauliOp.from_list(final_hamiltonian_entries)
"""

#size = 4
#J = [-1,0,0]
#h = 2
#full_final = spin_chain_improved.recursive_hamiltonian_sparse(size, J, h)
#full_final = spin_chain_improved.connected_chain_hamiltonian(size, J, h)
##full_final = full_final.toarray()
#final_hamiltonian = SparsePauliOp.from_operator(Operator(full_final))
#print(final_hamiltonian)
#print("remove sys.exit() to continue")
#sys.exit()

# Time evolve a given bit for time dt with magnetic field strength h pointing on z axis.
def magnetic_ev(q_c, bits, h, dt):
    q_c.rz(2*dt*h,bits)

# Convert a state (ordered in reverse) to full N^2 state vector
def convert_to_statevec(state):
    res = 0
    if state[-1] == "0":
        res = [1,0]
    else:
        res = [0,1]

    for i in range(len(state)-2,-1,-1):
        if state[i] == "0":
            res = np.kron(res, [1,0])
        else:
            res = np.kron(res, [0,1])

    print(res)
    return res

# returns <observable> = <state|observable|state>
def expectation_value(observable, state):
    return state.conj() @ (observable @ state)


def adiabatic_evolution(size, show_circuit, total_time, trotter_steps, J, H, backend, shots):
    dt = total_time/trotter_steps

    #full_final = spin_chain_improved.recursive_hamiltonian_sparse(size, J, H)
    #full_final = full_final.toarray()
    #final_hamiltonian = SparsePauliOp.from_operator(Operator(full_final))
    final_hamiltonian_entries = []
    for i in range(SIZE):
        string_of_ops = "I"*i+"Z"+"I"*(SIZE-i-1)
        if CONSTANT_H != 0:
            final_hamiltonian_entries.append((string_of_ops,-CONSTANT_H))

    for i in range(SIZE-1):
        string_of_ops_X = "I"*i + "XX" + "I"*(SIZE-i-2)
        string_of_ops_Y = "I"*i + "YY" + "I"*(SIZE-i-2)
        string_of_ops_Z = "I"*i + "ZZ" + "I"*(SIZE-i-2)
        if J[0] != 0:
            final_hamiltonian_entries.append((string_of_ops_X,-J[0]))
        if J[1] != 0:
            final_hamiltonian_entries.append((string_of_ops_Y,-J[1]))
        if J[2] != 0:
            final_hamiltonian_entries.append((string_of_ops_Z,-J[2]))

    final_hamiltonian_entries.append(("X" + "I"*(SIZE-2) + "X", -J[0]))
    final_hamiltonian_entries.append(("Y" + "I"*(SIZE-2) + "Y", -J[1]))
    final_hamiltonian_entries.append(("Z" + "I"*(SIZE-2) + "Z", -J[2]))

    final_hamiltonian = SparsePauliOp.from_list(final_hamiltonian_entries)
    print(final_hamiltonian)

    qc = QuantumCircuit(size,size)


    # Initial state preparation, default to |0>,
    # apply X to set to |1>
    #qc.x(0)
    #qc.x(1)
    #qc.x(2)
    #qc.x(3)

    # Actual adiabatic evolution procedure
    for s in np.linspace(0,1,trotter_steps):
        # TODO
        """TODO"""
        # Figure out why we need to include the -1 coefficient in from of the J
        # This might be due to a typo in the paper that presented the N gate?
        #full_N(qc, [0,1], -1*np.array(J), dt)
        #full_N(qc, [1,2], -1*s*np.array(J), dt)
        #full_N(qc, [2,3], -1*np.array(J), dt)
        for i in range(size):
            if i % 2 == 0:
                full_N(qc, [i,i+1], -1*np.array(J), dt)
            else:
                try:
                    full_N(qc, [i,i+1], -1*s*np.array(J), dt)
                    print(i,i+1)
                except:
                    full_N(qc, [i,0], -1*s*np.array(J), dt)
                    print("errored")
                    print(i,0)


        # Magnetically evolve all qubits
        for bit_num in range(size):
            magnetic_ev(qc, bit_num, H, dt)

    if show_circuit:
        qc.draw(output="mpl")
        plt.show()


    # Transpilation
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuit = pm.run(qc)

    isa_observables = final_hamiltonian.apply_layout(isa_circuit.layout)

    estimator = BackendEstimator(backend = backend)
    estimator.options.default_shots = shots

    
    # calculate [ <psi|hamiltonian|psi> ]
    job = estimator.run([(isa_circuit, isa_observables)])

    pub_result = job.result()[0]
    return pub_result.data.evs

if __name__ == "__main__":
    SIZE = 26 # Number of qubits
    SHOW_CIRCUIT = False
    TOTAL_TIME = 1      
    TROTTER_STEPS = 5
    D_TIME = TOTAL_TIME/TROTTER_STEPS 
    CONSTANT_H = 2
    J_MAG = [-1,0,0]
    SHOTS = 2048

    service = QiskitRuntimeService()
    #BACKEND = service.least_busy(operational=True, simulator=False, min_num_qubits=10)
    #BACKEND = FakeAlmadenV2()
    #BACKEND = FakeLagosV2()
    BACKEND = AerSimulator() # gives no noise, only statistical error associated with making measurements on quantum states.

    energy_expectation_value = adiabatic_evolution(SIZE, SHOW_CIRCUIT, TOTAL_TIME, TROTTER_STEPS, J_MAG, CONSTANT_H, BACKEND, SHOTS)
    print(f"Calculated <E> on {BACKEND}: {energy_expectation_value}")





"""
############################# OLD CODE USING SAMPLER BELOW

qc.measure(0,0)
qc.measure(1,1)
qc.measure(2,2)
qc.measure(3,3)
isa_circuit = pm.run(qc)

sampler = BackendSampler(backend = BACKEND)

job = sampler.run([isa_circuit],shots=SHOTS)
print(f"Job: {job.job_id()}     Status: {job.status()}")

result = job.result()

#print(f"!!!! {sampled_expectation_value(result.quasi_dists[0],final_hamiltonian)}")
#TODO: REMOVE HERE EXIT


dict_of_res = result.quasi_dists[0].binary_probabilities() #[0]
full_statevector = np.array([0]*SIZE**2, dtype=float)
for keyy in dict_of_res.keys():
    ## FIX HERE
    # how to make sure that the revlative phase is not lost?
    full_statevector += convert_to_statevec(keyy) * np.sqrt(dict_of_res[keyy])


print(full_statevector)

# probability of measuring 1 for each qubit, ordered in reverse ie [-1] = qubit 0
#prob_of_one = [0]*SIZE
#for i in range(SIZE):
#    for keyy in dict_of_res.keys():
#        if keyy[i] == "1":
#            prob_of_one[i] += dict_of_res[keyy]
#
#individual_states = []
#for prob in reversed(prob_of_one):
#    c0 = np.sqrt(prob)
#    c1 = np.sqrt(1-prob)
#    individual_states.append([c0, c1])
#
#final_statevector = individual_states[0] 
#for individual_state in individual_states[1:]:
#    final_statevector = np.kron(final_statevector, individual_state)
#
#print(final_statevector)
    
print(f"Counts for the meas output register: {dict_of_res}")
plt.bar(dict_of_res.keys(), dict_of_res.values(), 1, color='g')
plt.show()
if False:
    print("\n"*10)



    gened_hams = trotterization_exact.generate_initial_and_final_hamiltonians([2,2], J, CONSTANT_H)
    for a in gened_hams:
        print(a.toarray())
        import scipy
        print(scipy.sparse.linalg.eigsh(a))
        print(trotterization_exact.minimal_eigenvector(a))


    print("Calculating overlap: ")
    print(gened_hams[0])

    test_state = convert_to_statevec("0000")
    print(f"ground state overlap {np.matmul(test_state.conj(), (gened_hams[0] @ test_state))} ")
        
    test_2 = spin_chain_improved.recursive_hamiltonian_sparse(2, J, 3)
    print(test_2.toarray())
    import scipy
    print(scipy.linalg.eigh(test_2.toarray()))

    print("\n"*10)

# calculting the expectation value of the hamiltonian 
# based on the actual matrix representation

full_final = spin_chain_improved.recursive_hamiltonian_sparse(SIZE, J, CONSTANT_H)
full_final = full_final.toarray()
#print(full_final)
import scipy
#print(scipy.linalg.eigh(full_final))
test_state = full_statevector
print(full_final)
print(f"resulting state: {test_state}")
#print(f"<E> {expectation_value(full_final, test_state)}")
"""

"""
# Experiemntal...
#
# 99% does not work
list_to_sum = []
for key in dict_of_res:
    converted_s = convert_to_statevec(key)
    energy_exp_single = expectation_value(full_final, converted_s)
    print(f"Energy: {int(energy_exp_single)}")

    list_to_sum.append(dict_of_res[key]/SHOTS * energy_exp_single)

print(f"Expectation value of energy: {sum(list_to_sum)}")
"""
"""
final_hamiltonian_entries = []
for i in range(SIZE):
    string_of_ops = "I"*i+"Z"+"I"*(SIZE-i-1)
    if CONSTANT_H != 0:
        final_hamiltonian_entries.append((string_of_ops,-CONSTANT_H))

for i in range(SIZE-1):
    string_of_ops_X = "I"*i + "XX" + "I"*(SIZE-i-2)
    string_of_ops_Y = "I"*i + "YY" + "I"*(SIZE-i-2)
    string_of_ops_Z = "I"*i + "ZZ" + "I"*(SIZE-i-2)
    if J[0] != 0:
        final_hamiltonian_entries.append((string_of_ops_X,-J[0]))
    if J[1] != 0:
        final_hamiltonian_entries.append((string_of_ops_Y,-J[1]))
    if J[2] != 0:
        final_hamiltonian_entries.append((string_of_ops_Z,-J[2]))

final_hamiltonian = SparsePauliOp.from_list(final_hamiltonian_entries)
print(final_hamiltonian)
"""



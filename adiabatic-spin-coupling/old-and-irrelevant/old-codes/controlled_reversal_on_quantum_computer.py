import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, Gate
from qiskit_aer.primitives import Sampler
from matplotlib import pyplot as plt
from qiskit_aer import AerSimulator

# For adiabatic evolution
from spin_fusion_on_quantum_computer import adiabatic_evolution_circuit

# could include alternating XY pauli gate as controlled reversal gate

def XX_interaction(L, beta, cxx, qc):
    # Horizontal interactions
    for i in range(L):
        qc.h(i)
        qc.h(i + 1)
        qc.cx(i, i + 1)
        qc.rz(cxx * beta, i + 1)
        qc.cx(i, i + 1)

# Transverse magnetic field
def magn_interaction(L, beta, hz, qc):
    for i in range(L):
        qc.rz(2 * hz * beta, i) # Factor of 2 since that is how we defined the Second-Order Trotterization
"""
def rodeo_cycle(L, t: Parameter, r, cxx, hz, targ: Parameter, beta):
    #beta = t / r # delta T for trotter steps
    r = t/beta
    sys = QuantumRegister(L, 's')
    aux = QuantumRegister(1, 'a')
    qc = QuantumCircuit(sys, aux)

    # Initialize ancilla qubit to be in superposition
    qc.h(aux[0])

    # Add Y reversal gates to odd, X reversal gates to even qubits
    for i in range(L):
        idx = i
        if (i) % 2 == 0:
            qc.cx(aux[0], sys[idx])
        else:
            qc.cy(aux[0], sys[idx])

    # Trotter time evolution
    for _ in range(r):
        XX_interaction(L, beta, cxx, qc)
        #YY_interaction(N, M, beta, cyy, qc)
        magn_interaction(L, beta, hz, qc)
        #YY_interaction(N, M, beta, cyy, qc)
        XX_interaction(L, beta, cxx, qc)

    # Add reversal gates at end
    for i in range(L):
        idx = i
        if (i) % 2 == 0:
            qc.cx(aux[0], sys[idx])
        else:
            qc.cy(aux[0], sys[idx])
            
    # Add phase gate to ancilla
    qc.p(2*targ * t, aux[0])

    # Revert superposition
    qc.h(aux[0])

    return qc.to_gate(label=f"Rodeo_Cycle_{r}_Trotter_Steps")
"""

def rodeo_cycle(L, t, r, cxx, hz, targ: Parameter, beta):
    beta = t / r # delta T for trotter steps
    #r = int(t/beta)
    sys = QuantumRegister(L, 's')
    aux = QuantumRegister(1, 'a')
    qc = QuantumCircuit(sys, aux)

    # Initialize ancilla qubit to be in superposition
    qc.h(aux[0])

    # Add Y reversal gates to odd, X reversal gates to even qubits
    for i in range(L):
        idx = i
        if (i) % 2 == 0:
            qc.cx(aux[0], sys[idx])
        else:
            qc.cy(aux[0], sys[idx])

    # Trotter time evolution
    for _ in range(r):
        XX_interaction(L, beta, cxx, qc)
        #YY_interaction(N, M, beta, cyy, qc)
        magn_interaction(L, beta, hz, qc)
        #YY_interaction(N, M, beta, cyy, qc)
        XX_interaction(L, beta, cxx, qc)

    # Add reversal gates at end
    for i in range(L):
        idx = i
        if (i) % 2 == 0:
            qc.cx(aux[0], sys[idx])
        else:
            qc.cy(aux[0], sys[idx])
            
    # Add phase gate to ancilla
    qc.p(2*targ * t, aux[0])

    # Revert superposition
    qc.h(aux[0])

    return qc.to_gate(label=f"Rodeo_Cycle_{r}_Trotter_Steps")

# Parameters
L = 2
cxx = 1
J = [-cxx, 0,0]
hz = 0.5
numqubits = L
cycles = 3
adiabatic_time = 1
trotter_steps = 3


# Initialize system parameters
sysqubits = 1
timeresamples = 10

# Create Target and t parameters
targ = Parameter(r'$E_\odot$')
#t = [Parameter(fr'$t_{i}$') for i in range(cycles)]
#r = 5
beta = 0.1

# Create a list of target energies at the same length of the cycle
targ_list = [targ] * cycles

# Create registers and circuit
classical = ClassicalRegister(cycles, 'c')
aux = QuantumRegister(1, 'a')
sys = QuantumRegister(numqubits, 's')
circuit = QuantumCircuit(classical, sys, aux)
#adiabatic_evolution_circuit(L, adiabatic_time, trotter_steps, J, hz, AerSimulator(method="matrix_product_state"), circuit, sys)

# circuit.x(0)
# initial_state
i_state = np.array([ 0.92387953+0.j, -0.        -0.j ,-0.        -0.j ,-0.38268343-0.j])
#i_state = np.array([0.92,0,0,-0.38])
i_state = i_state/np.linalg.norm(i_state)
circuit.initialize(i_state, [0,1])

gamma = 1/2
for _ in range(timeresamples):
    tsamples = ((1 / gamma) * np.abs(np.random.randn(cycles))).tolist()
    #tsamples=np.ones(cycles)
    #parameter_binds.append({t[i]: tsamples[i] for i in range(cycles)})

print(tsamples)

#parameter_binds = zip(t, tsamples)
#parameters = dict(parameter_binds)




# Create circuit
for j in range(cycles):
    circuit.append(rodeo_cycle(L, tsamples[j], 0, cxx, hz, targ_list[j],beta), range(1 + numqubits))
    circuit.measure(aux, classical[j])

print(circuit)
#circuit.draw(output="mpl")
#plt.show()

target = {targ : -3.3}
#circuit1 = circuit.assign_parameters(parameters, inplace =False)
#circuit2.draw(output= 'mpl')
circuit2 = circuit.assign_parameters(target, inplace = False)

sampler = Sampler()

job = sampler.run(circuit2)

job.result()

quasi_dists = job.result().quasi_dists

print(quasi_dists)



# Enumerate scan energies
energymin = -4
energymax = 4
stepsize = 0.1

targetenergies = np.linspace(energymin, energymax, int((energymax-energymin)/stepsize))
targetenergynum = len(targetenergies)
print("Number of target energies:", targetenergynum)

#Energy window, which should to be slightly larger than stepsize in scan
# Is inverse of sigma parameter

# Amount of "scrambling" of t per target energy. The more random the t the better. 
timeresamples = 10 # Resampling of times for given target energy
shots_per_same_time = 1024

# Create empty list for data
data = []

# Loop through energy list
for i in range(len(targetenergies)):
    
    # Creates dictionary for target energy parameter
    targ_energy = {targ : targetenergies[i]}

    # Below is a troubleshooting line you can use to see if the code is scanning through energies properly
    print("Executing for Target Energy:", targ_energy)
    
    # Initialize a list that will contain the results of all 10 resamples for 1 target energy
    targetenergyrun = []
    for _ in range(timeresamples):
        # Creates random time samples for 1 run
        tsamples = ((1 / gamma) * np.random.randn(cycles)).tolist()
        
        # Creates a dictionary to be able to bind time samples to time parameters
        #time_parameter_binds = zip(t, tsamples)
        #time_parameters = dict(time_parameter_binds)
        
        # Assigns target energy and time values to parameters
        #circuit1 = circuit.assign_parameters(time_parameters, inplace =False)
        circuit2 = circuit.assign_parameters(targ_energy, inplace = False)
        
        # Runs simulation of circuit with values
        sampler = Sampler()
        job = sampler.run(circuit2)
        job.result()
        quasi_dists = job.result().quasi_dists
        
        # Appends the results to list for this target energy
        targetenergyrun.append(quasi_dists)
    
    # The output from above needs to be post-processed as shown below to gain meaning from it:

    # Flattens list of list of dictionaries into just a list of dictionaries
    flattened_list = []
    for sublist in targetenergyrun:
        flattened_list.extend(sublist)

    # Sums and average dictionaries from multiple timeresamples
    combined_dict = {} 
    for dictionary in flattened_list:
        for key, value in dictionary.items():
            combined_dict[key] = combined_dict.get(key, 0) + value

    average_dict = {}
    for key in combined_dict:
        average_dict[key] = combined_dict[key] / timeresamples

    data.append(average_dict)


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# This extracts the probabilities for the 0 bitcounts from our obtained data
values_list = []
for d in data:
    if 0 in d:
        values_list.append(d[0])
    else:
        values_list.append(0.0)
print(len(values_list))
print(len(targetenergies))

# Define a Gaussian function
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

# Define a sum of multiple Gaussians
def sum_of_gaussians(x, *params):
    n_gaussians = len(params) // 3
    result = np.zeros_like(x)
    for i in range(n_gaussians):
        amp = params[i*3]
        cen = params[i*3 + 1]
        wid = params[i*3 + 2]
        result += gaussian(x, amp, cen, wid)
    return result

# Initial guess for the parameters: amplitudes, centers, and widths of the Gaussians
initial_guess = [
    1, -13, 1
]

# Fit the sum of Gaussians to the data
#popt, pcov = curve_fit(sum_of_gaussians, targetenergies, values_list, p0=initial_guess)

# Extract the fitted parameters
#fitted_params = popt

# Generate x values for plotting the fit
#x_fit = np.linspace(min(targetenergies), max(targetenergies), 1000)
#y_fit = sum_of_gaussians(x_fit, *fitted_params)

# Plot the data and the fit
plt.scatter(targetenergies, values_list, label='Data')
#plt.plot(x_fit, y_fit, color='red', label='Sum of Gaussians fit')
plt.grid()
plt.xlabel('Target Energy')
plt.ylabel('Average Probability')
#plt.axvline(x = -12.53, color = 'green', label = 'Ground State Energy')
plt.legend()
plt.show()

# Print the peaks (centers) of the Gaussians
#n_gaussians = len(fitted_params) // 3
#peaks = [fitted_params[i*3 + 1] for i in range(n_gaussians)]
#print(f'The peaks (centers) of the Gaussians are at: {peaks}')

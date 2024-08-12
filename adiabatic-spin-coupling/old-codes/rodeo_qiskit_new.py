import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, Gate
from qiskit_aer.primitives import Sampler

def XX_interaction(N, M, beta, cxx, qc):
    # Horizontal interactions
    for row in range(N):
        for col in range(M - 1):
            i = row * M + col
            qc.h(i)
            qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(cxx * beta, i + 1)
            qc.cx(i, i + 1)
            qc.h(i)
            qc.h(i + 1)

def YY_interaction(N, M, beta, cyy, qc):
    # Horizontal interactions
    for row in range(N):
        for col in range(M - 1):
            i = row * M + col
            qc.rx(-np.pi/2, i)
            qc.rx(-np.pi/2, i + 1)
            qc.cx(i, i + 1)
            qc.rz(cyy * beta, i + 1)
            qc.cx(i, i + 1)
            qc.rx(np.pi/2, i)
            qc.rx(np.pi/2, i + 1)

def magn_interaction(N, M, beta, hz, qc):
    # Transverse magnetic field
    for i in range(N * M):
        qc.rz(2 * hz * beta, i) # Factor of 2 since that is how we defined the Second-Order Trotterization

def rodeo_cycle(N, M, t: Parameter, r, cxx, cyy, hz, targ: Parameter):
    beta = t / r # Parameter for rotation gates
    sys = QuantumRegister(N * M, 's')
    aux = QuantumRegister(1, 'a')
    qc = QuantumCircuit(sys, aux)

    # Initialize ancilla qubit to be in superposition
    qc.h(aux[0])

    # Add Y reversal gates to odd, X reversal gates to even qubits
    for row in range(N):
        for col in range(M):
            idx = row * M + col
            if (row + col) % 2 == 0:
                qc.cx(aux[0], sys[idx])
            else:
                qc.cy(aux[0], sys[idx])

    # Trotter time evolution
    for _ in range(r):
        XX_interaction(N, M, beta, cxx, qc)
        YY_interaction(N, M, beta, cyy, qc)
        magn_interaction(N, M, beta, hz, qc)
        YY_interaction(N, M, beta, cyy, qc)
        XX_interaction(N, M, beta, cxx, qc)

    # Add reversal gates at end
    for row in range(N):
        for col in range(M):
            idx = row * M + col
            if (row + col) % 2 == 0:
                qc.cx(aux[0], sys[idx])
            else:
                qc.cy(aux[0], sys[idx])
                
    # Add phase gate to ancilla
    qc.p(2*targ * t, aux[0])

    # Revert superposition
    qc.h(aux[0])
    return qc.to_gate(label=f"Rodeo_Cycle_{r}_Trotter_Steps")
def run(cycles,verification_cycles,tsamples1, tsamples2, sigma, sigma2):
    # Parameters
    N = 1 # Keep at 1 for a 1-dimensional chain
    M = 4
    cxx = -1  # Jx
    cyy = -1  # Jy
    hz = 0
    numqubits = N * M
    verification_cycles += cycles # TOTAL NUMBER OF CYCLES, INCORRECTLY NAMED

    # Create Target and t parameters
    targ = Parameter(r'$E_\odot$')
    t1 = [Parameter(fr'$t_{i}$') for i in range(cycles)]
    t2 = [Parameter(fr'$tt_{j}$') for j in range(cycles,verification_cycles)]
    r = 10 # Trotter steps

    # Create a list of target energies at the same length of the cycle
    targ_list = [targ] * (cycles+verification_cycles)

    # Create registers and circuit
    classical = ClassicalRegister(verification_cycles, 'c')
    aux = QuantumRegister(1, 'a')
    sys = QuantumRegister(numqubits, 's')
    circuit = QuantumCircuit(classical, sys, aux)

    i_state = np.array([0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0. , 0. , 0.5, 0.5, 0. , 0. , 0. , 0. , 0. ])
    #i_state = np.array([0.00000000e+00+0.00000000e+00j,4.82622160e-17+1.41429746e-47j,
    # 1.47240917e-16+3.69841314e-48j,2.23606798e-01-4.19967075e-33j,
    # 1.33115188e-16-6.23115354e-48j,5.00000000e-01-2.14293049e-33j,
    # 4.47213595e-01-2.25107346e-33j,0.00000000e+00+0.00000000e+00j,
    # 2.66483202e-16-2.97752699e-48j,4.47213595e-01-1.84719475e-33j,
    # 5.00000000e-01+0.00000000e+00j,0.00000000e+00+0.00000000e+00j,
    # 2.23606798e-01+1.28951937e-34j,0.00000000e+00+0.00000000e+00j,
    # 0.00000000e+00+0.00000000e+00j,0.00000000e+00+0.00000000e+00j])
    #i_state = np.array([ 0.00000000e+00-0.00000000e+00j,-4.82622160e-17+1.12132209e-47j,
    #  1.92720982e-16+1.95020099e-48j,-2.23606798e-01+1.01695967e-33j,
    # -1.33115188e-16-1.78140064e-47j, 5.00000000e-01-8.44720052e-33j,
    # -4.47213595e-01-2.71039304e-34j, 0.00000000e+00-0.00000000e+00j,
    #  2.21003138e-16-2.64676764e-48j,-4.47213595e-01-1.32818945e-33j,
    #  5.00000000e-01+0.00000000e+00j, 0.00000000e+00-0.00000000e+00j,
    # -2.23606798e-01-1.08655441e-33j, 0.00000000e+00-0.00000000e+00j,
    #  0.00000000e+00-0.00000000e+00j, 0.00000000e+00-0.00000000e+00j])
    #i_state = np.array([-0.        -0.j,-0.        -0.j,-0.        -0.j,-0.        -0.j,
    #    -0.        -0.j,-0.        -0.j,-0.        -0.j, 0.37174803-0.j,
    # -0.        -0.j,-0.        -0.j,-0.        -0.j,-0.60150096-0.j,
    # -0.        -0.j, 0.60150096+0.j,-0.37174803-0.j,-0.        -0.j])
    #i_state = np.array([0,1,-1,0])
    """
    i_state = np.array([0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 2.32924149e-33+1.36514259e-63j,
        7.10617294e-33+2.26091816e-63j, 1.07917596e-17+2.95977985e-48j,
        6.42443396e-33+1.58191544e-63j, 2.41311080e-17+6.96806473e-48j,
        2.15835191e-17+6.21628872e-48j, 0.00000000e+00+0.00000000e+00j,
        1.28610699e-32+3.62516311e-63j, 2.15835191e-17+6.23578080e-48j,
        2.41311080e-17+7.07148730e-48j, 0.00000000e+00+0.00000000e+00j,
        1.07917596e-17+3.16868877e-48j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 7.10617294e-33+2.26091816e-63j,
        2.16798876e-32+1.08911548e-63j, 3.29240700e-17+2.08626948e-49j,
        1.96000023e-32-4.25165801e-64j, 7.36204585e-17+1.53367952e-48j,
        6.58481398e-17+1.32253052e-48j, 0.00000000e+00+0.00000000e+00j,
        3.92372310e-32+5.47151171e-64j, 6.58481398e-17+1.38199799e-48j,
        7.36204585e-17+1.84920657e-48j, 0.00000000e+00+0.00000000e+00j,
        3.29240700e-17+8.45977321e-49j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 1.07917596e-17+2.95977985e-48j,
        3.29240700e-17+2.08626948e-49j, 5.00000001e-02-1.87814986e-33j,
        2.97654610e-17-1.95236825e-48j, 1.11803399e-01-2.57900920e-33j,
        1.00000000e-01-2.38150518e-33j, 0.00000000e+00+0.00000000e+00j,
        5.95874555e-17-1.78493698e-48j, 1.00000000e-01-2.29119516e-33j,
        1.11803399e-01-2.09983537e-33j, 0.00000000e+00+0.00000000e+00j,
        5.00000001e-02-9.10240399e-34j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 6.42443396e-33+1.58191544e-63j,
        1.96000023e-32-4.25165801e-64j, 2.97654610e-17-1.95236825e-48j,
        1.77196533e-32-1.65892235e-63j, 6.65575940e-17-3.40083337e-48j,
        5.95309218e-17-3.08630864e-48j, 0.00000000e+00+0.00000000e+00j,
        3.54729615e-32-2.05685181e-63j, 5.95309218e-17-3.03254625e-48j,
        6.65575940e-17-3.11557677e-48j, 0.00000000e+00+0.00000000e+00j,
        2.97654610e-17-1.37616283e-48j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 2.41311080e-17+6.96806473e-48j,
        7.36204585e-17+1.53367952e-48j, 1.11803399e-01-2.57900920e-33j,
        6.65575940e-17-3.40083337e-48j, 2.50000000e-01-2.14293049e-33j,
        2.23606798e-01-2.08388438e-33j, 0.00000000e+00+0.00000000e+00j,
        1.33241601e-16-2.05981847e-48j, 2.23606798e-01-1.88194502e-33j,
        2.50000000e-01-1.07146524e-33j, 0.00000000e+00+0.00000000e+00j,
        1.11803399e-01-4.14697857e-34j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 2.15835191e-17+6.21628872e-48j,
        6.58481398e-17+1.32253052e-48j, 1.00000000e-01-2.38150518e-33j,
        5.95309218e-17-3.08630864e-48j, 2.23606798e-01-2.08388438e-33j,
        2.00000000e-01-2.01342131e-33j, 0.00000000e+00+0.00000000e+00j,
        1.19174911e-16-1.93146381e-48j, 2.00000000e-01-1.83280126e-33j,
        2.23606798e-01-1.12553673e-33j, 0.00000000e+00+0.00000000e+00j,
        1.00000000e-01-4.45686269e-34j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 1.28610699e-32+3.62516311e-63j,
        3.92372310e-32+5.47151171e-64j, 5.95874555e-17-1.78493698e-48j,
        3.54729615e-32-2.05685181e-63j, 1.33241601e-16-2.05981847e-48j,
        1.19174911e-16-1.93146381e-48j, 0.00000000e+00+0.00000000e+00j,
        7.10132969e-32-1.58692185e-63j, 1.19174911e-16-1.82383692e-48j,
        1.33241601e-16-1.48876349e-48j, 0.00000000e+00+0.00000000e+00j,
        5.95874555e-17-6.31431751e-49j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 2.15835191e-17+6.23578080e-48j,
        6.58481398e-17+1.38199799e-48j, 1.00000000e-01-2.29119516e-33j,
        5.95309218e-17-3.03254625e-48j, 2.23606798e-01-1.88194502e-33j,
        2.00000000e-01-1.83280126e-33j, 0.00000000e+00+0.00000000e+00j,
        1.19174911e-16-1.82383692e-48j, 2.00000000e-01-1.65218121e-33j,
        2.23606798e-01-9.23597375e-34j, 0.00000000e+00+0.00000000e+00j,
        1.00000000e-01-3.55376244e-34j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 2.41311080e-17+7.07148730e-48j,
        7.36204585e-17+1.84920657e-48j, 1.11803399e-01-2.09983537e-33j,
        6.65575940e-17-3.11557677e-48j, 2.50000000e-01-1.07146524e-33j,
        2.23606798e-01-1.12553673e-33j, 0.00000000e+00+0.00000000e+00j,
        1.33241601e-16-1.48876349e-48j, 2.23606798e-01-9.23597375e-34j,
        2.50000000e-01+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        1.11803399e-01+6.44759685e-35j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 1.07917596e-17+3.16868877e-48j,
        3.29240700e-17+8.45977321e-49j, 5.00000001e-02-9.10240399e-34j,
        2.97654610e-17-1.37616283e-48j, 1.11803399e-01-4.14697857e-34j,
        1.00000000e-01-4.45686269e-34j, 0.00000000e+00+0.00000000e+00j,
        5.95874555e-17-6.31431751e-49j, 1.00000000e-01-3.55376244e-34j,
        1.11803399e-01+6.44759685e-35j, 0.00000000e+00+0.00000000e+00j,
        5.00000001e-02+5.76690595e-35j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j,
        0.00000000e+00+0.00000000e+00j, 0.00000000e+00+0.00000000e+00j])
    """
    i_state = i_state/np.linalg.norm(i_state)
    circuit.initialize(i_state, [0,1,2,3])
    #circuit.initialize(i_state, [0,1,2,3,4,5,6,7])
    #circuit.initialize(i_state, [0,1])

    #circuit.x(0)
    #circuit.x(1)

    # Create circuit
    for j in range(cycles):
        circuit.append(rodeo_cycle(N, M, t1[j], r, cxx, cyy, hz, targ_list[j]), range(1 + numqubits))
        circuit.measure(aux, classical[j])

    for j in range(cycles, verification_cycles):
        circuit.append(rodeo_cycle(N, M, t2[j-cycles], 20, cxx, cyy, hz, targ_list[j]), range(1 + numqubits))
        circuit.measure(aux, classical[j])



    #tsamples = (sigma * np.abs(np.random.randn(cycles))).tolist()
    #    
    #parameter_binds = zip(t, tsamples)
    #parameters = dict(parameter_binds)
    #
    #target = {targ : -12.5}
    #    
    #circuit1 = circuit.assign_parameters(parameters, inplace =False)
    #circuit2 = circuit1.assign_parameters(target, inplace = False)

    #sampler = Sampler()
    #
    #job = sampler.run(circuit2)
    #
    #job.result()
    #
    #quasi_dists = job.result().quasi_dists
    #
    #print(quasi_dists)
    #
    # Enumerate scan energies
    #energymin = 4.35
    #energymax = 4.55
    #tepsize = 0.01

    #energymin = 4.4
    #energymax = 4.55
    #stepsize = 0.01

    energymin=3
    energymax = 6
    stepsize=0.1


    #energymin = 4.435
    #energymax =4.46
    #stepsize = 0.005

    #energymin = 0
    #energymax = 20
    #stepsize = 2

    targetenergies = np.linspace(energymin, energymax, int((energymax-energymin)/stepsize))
    targetenergynum = len(targetenergies)
    #print("Number of target energies:", targetenergynum)

    #Energy window, which should to be slightly larger than stepsize in scan
    # Is inverse of sigma parameter

    # Amount of "scrambling" of t per target energy. The more random the t the better. 
    timeresamples = 1 # Resampling of times for given target energy

    #Create empty list for data
    data = []
    data_sd = []
    data2 = []
    data2_sd = []

    # Loop through energy list
    #tsamples1 = (sigma * np.random.randn(cycles)).tolist()
    #tsamples2 = (sigma2 * np.random.randn(verification_cycles)).tolist()
    for i in range(len(targetenergies)):
        # Creates dictionary for target energy parameter
        targ_energy = {targ : targetenergies[i]}

        # Below is a troubleshooting line you can use to see if the code is scanning through energies properly
        #print("Executing for Target Energy:", targ_energy)
        
        # Initialize a list that will contain the results of all 10 resamples for 1 target energy
        targetenergyrun = []
        targetenergyrun2 = []
        for _ in range(timeresamples):
            # Creates random time samples for 1 run
            #tsamples1 = (sigma * np.random.randn(cycles)).tolist()
            #tsamples2 = (sigma2 * np.random.randn(verification_cycles)).tolist()
            
            # Creates a dictionary to be able to bind time samples to time parameters
            time_parameter_binds1 = zip(t1, tsamples1)
            time_parameter_binds2 = zip(t2, tsamples2)
            time_parameters1 = dict(time_parameter_binds1)
            time_parameters2 = dict(time_parameter_binds2)
            
            # Assigns target energy and time values to parameters
            circuit1 = circuit.assign_parameters(time_parameters1, inplace =False)
            circuit2 = circuit1.assign_parameters(time_parameters2, inplace =False)
            circuit3 = circuit2.assign_parameters(targ_energy, inplace = False)
            #circuit3.draw("mpl")
            
            # Runs simulation of circuit with values
            sampler = Sampler()#run_options = {'shots':100000})
            job = sampler.run(circuit3)
            #print(job.result())
            quasi_dists = job.result().quasi_dists[0].binary_probabilities()

            numerator = 0
            denominator = 0

            #print(quasi_dists)
            for bin_val in quasi_dists.keys():
                if bin_val[-cycles:] == "0"*cycles:
                    denominator += quasi_dists[bin_val]

                #if bin_val[:cycles] == "0"*cycles:
                #    denominator += quasi_dists[bin_val]
                if bin_val == "0"*(verification_cycles):
                    numerator += quasi_dists[bin_val]

            targetenergyrun.append(denominator)
            targetenergyrun2.append(numerator/denominator)
            #print(f"time samples: {tsamples2}")
            #print(f"successful chance first rodeo: {denominator}")
            #print(f"successful chance second rodeo given successful first rodeo: {numerator/denominator}")

            # Appends the results to list for this target energy
            #targetenergyrun.append(quasi_dists)
        
        # The output from above needs to be post-processed as shown below to gain meaning from it:

        # Flattens list of list of dictionaries into just a list of dictionaries
        #flattened_list = []
        #for sublist in targetenergyrun:
        #    flattened_list.extend(sublist)

        # Sums and average dictionaries from multiple timeresamples
        #combined_dict = {} 
        #for dictionary in flattened_list:
        #    for key, value in dictionary.items():
        #        combined_dict[key] = combined_dict.get(key, 0) + value

        #average_dict = {}
        #for key in combined_dict:
        #    average_dict[key] = combined_dict[key] / timeresamples

        #data.append(average_dict)
        data.append(np.average(targetenergyrun))
        data_sd.append(np.std(targetenergyrun))
        data2.append(np.average(targetenergyrun2))
        data2_sd.append(np.std(targetenergyrun2))

    import matplotlib.pyplot as plt
    plt.errorbar(targetenergies, data, data_sd, linestyle='None', marker='^')
    plt.show()
    plt.errorbar(targetenergies, data2, data2_sd)
    plt.show()
    print(f"Average success probability (2nd RA): {np.sum(data2)/len(data2)}")
    print(f"Max success probability (2nd RA): {np.max(data2)}")
    #return np.sum(data2)/len(data2)
    return np.max(data2)

    def to_writable(lis):
        to_write = ""
        for item in lis:
            to_write += str(item)
            to_write += ","
        return to_write[:-1]

    with open("output_rodeo_qiskit_new.txt", "a") as f:
        f.write(to_writable(targetenergies))
        f.write("\n")
        f.write(to_writable(data))
        f.write("\n")
        f.write(to_writable(data_sd))
        f.write("\n")

if __name__ == "__main__":
    verification_cycles = 0
    sigma = 3
    sigma2 = 3
    for i in range(10):
        tsamples2 = (sigma2 * np.random.randn(verification_cycles)).tolist()
        cy_list = [1,2,3,4,5,10,15]
        tsamples1_l = [(sigma*np.random.randn(cy_list[0])).tolist()]
        for j in range(1,len(cy_list)):
            new_v = (sigma*np.random.randn(cy_list[j]-cy_list[j-1])).tolist()
            tsamples1_l.append(tsamples1_l[j-1]+new_v)


        res_str = ""
        for j in range(len(cy_list)):
            cy_num = cy_list[j]
            #tsamples1 = (sigma * np.random.randn(cycles)).tolist()
            tsamples1 = tsamples1_l[j]
            res_str += str(run(cy_num,verification_cycles,tsamples1, tsamples2, sigma, sigma2)) 
            res_str += ","
        print(res_str[:-1])

    """
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
    initial_guess = [1, -1.41, 1]

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

    """
# Print the peaks (centers) of the Gaussians
#n_gaussians = len(fitted_params) // 3
#peaks = [fitted_params[i*3 + 1] for i in range(n_gaussians)]
#print(f'The peaks (centers) of the Gaussians are at: {peaks}')

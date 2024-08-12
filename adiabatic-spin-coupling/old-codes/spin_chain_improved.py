# Code for generating the hamiltonian of Quantum Heisenberg chain
# Uses sparse matricies and a recursive method to achieve optimal runtime

import numpy as np
import scipy

# pauli matricies 
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])
iden = np.array([[1,0],[0,1]])

# Sloppy implementation of extending a 1-qubit operator (mat_inner)
# To apply to (dimensions) qubits.
# Generates the resulting operator for each position that the given mat_inner could take.
# Helper function to old generator, hamiltonian_of_set.
def gen_j_matricies(dimensions, mat_inner):
    l = []
    for j in range(dimensions):
        mat = iden
        if j == 0:
            mat = mat_inner
        for i in range(j-1):
            mat = np.kron(mat, iden)
        if j != 0:
            mat = np.kron(mat, mat_inner)
        for i in range(dimensions-j-1):
            mat = np.kron(mat, iden)

        l.append(mat)

    return l

# Generates the hamiltonian of a Heisenberg model of a spin-1/2 chain
# Very slow, only used to generate 2-qubit for newer, recursive_hamiltonian function
def hamiltonian_of_set(size, J, h):
    Jx=J[0]
    Jy=J[1]
    Jz=J[2]

    s1_js = gen_j_matricies(size,s1)
    s2_js = gen_j_matricies(size,s2)
    s3_js = gen_j_matricies(size,s3)

    spin_magnet_hamiltonian = sum(s3_js)*h
    spin_coupling_hamiltonian = 1j*np.zeros((2**size, 2**size))

    for i in range(len(s1_js)-1):
        cx = np.matmul(s1_js[i],s1_js[i+1])
        cy = np.matmul(s2_js[i],s2_js[i+1])
        cz = np.matmul(s3_js[i],s3_js[i+1])
        spin_coupling_hamiltonian += Jx*cx+Jy*cy+Jz*cz

    return (-1)*(spin_magnet_hamiltonian + spin_coupling_hamiltonian)

def recursive_hamiltonian_step(H, N, J, h):
    new_hamiltonian = np.kron(H, np.identity(2))
    new_spin_term = np.kron(np.identity(2**(N-1)), J[0]*np.kron(s1, s1) + J[1]*np.kron(s2,s2) + J[2]*np.kron(s3,s3))
    new_magnetic_term = h * np.kron(np.identity(2**(N)), s3)

    return new_hamiltonian - new_spin_term - new_magnetic_term

def recursive_hamiltonian(N, J, h):
    initial = hamiltonian_of_set(2, J, h)
    remaining_steps = N-2
    while remaining_steps > 0:
        N_next = N - remaining_steps
        initial = recursive_hamiltonian_step(initial, N_next, J, h)
        remaining_steps -= 1

    return initial

# Newer recursive generator with scipy's sparse matricies
def recursive_hamiltonian_step_sparse(H, N, J, h):
    new_hamiltonian = scipy.sparse.kron(H,scipy.sparse.identity(2))
    new_spin_term = scipy.sparse.kron(scipy.sparse.identity(2**(N-1)), scipy.sparse.csr_array(J[0]*np.kron(s1, s1) + J[1]*np.kron(s2,s2) + J[2]*np.kron(s3,s3)))
    new_magnetic_term = h * scipy.sparse.kron(scipy.sparse.identity(2**(N)), scipy.sparse.csr_array(s3))

    return new_hamiltonian - new_spin_term - new_magnetic_term

def recursive_hamiltonian_sparse(N, J, h):
    initial = scipy.sparse.csr_array(hamiltonian_of_set(2, J, h))
    remaining_steps = N-2
    while remaining_steps > 0:
        N_next = N - remaining_steps
        initial = recursive_hamiltonian_step_sparse(initial, N_next, J, h)
        remaining_steps -= 1

    return initial

# Almost the same as the normal, recursive_hamiltonian_sparse, but with a connection between first and last qubit
def connected_chain_hamiltonian(N, J, h):
    unconnected_chain_hamiltonian = hamiltonian_of_set(N, J, h)
    if N == 2:
        return unconnected_chain_hamiltonian
    
    connection_spin_term_x = scipy.sparse.kron(s1, scipy.sparse.identity(2**(N-2)))
    connection_spin_term_x = scipy.sparse.kron(connection_spin_term_x, s1)
    connection_spin_term_x *= J[0]

    connection_spin_term_y = scipy.sparse.kron(s2, scipy.sparse.identity(2**(N-2)))
    connection_spin_term_y = scipy.sparse.kron(connection_spin_term_y, s2)
    connection_spin_term_y *= J[1]

    connection_spin_term_z = scipy.sparse.kron(s3, scipy.sparse.identity(2**(N-2)))
    connection_spin_term_z = scipy.sparse.kron(connection_spin_term_z, s3)
    connection_spin_term_z *= J[2]

    connection_spin_term = connection_spin_term_x + connection_spin_term_y + connection_spin_term_z
    connection_spin_term *= -1
    
    connected_chain_h = connection_spin_term + unconnected_chain_hamiltonian

    return connected_chain_h

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    #test_size = 24
    #t1 = time.perf_counter()
    #sparse_gen = recursive_hamiltonian_sparse(test_size,[-1,-1,-1],1)
    #t2 = time.perf_counter()
    #gen = recursive_hamiltonian(test_size,[-1,-1,-1],1)
    #t3 = time.perf_counter()

    #print(f"sparse time: {t2-t1}")

    #plt.show
    #print(sparse_gen)
    #print(gen)
    #print(np.array_equal(sparse_gen,gen))

    #print(np.array_equal(hamiltonian_of_set(4,[1,2,5],-1), recursive_hamiltonian(4,[1,2,5],-1)))
    #h3 = recursive_hamiltonian(hamiltonian_of_set(2, [1,1,1], 1),2,[1,1,1],1)
    #h4 = recursive_hamiltonian(h3, 3, [1,1,1],1)
    #h5 = recursive_hamiltonian(h4, 4, [1,1,1],1)
    #h6 = recursive_hamiltonian(h5, 5, [1,1,1],1)
    #h7 = recursive_hamiltonian(h6, 6, [1,1,1],1)
    #h8 = recursive_hamiltonian(h7, 7, [1,1,1],1)
    #h9 = recursive_hamiltonian(h8, 8, [1,1,1],1)
    #h10 = recursive_hamiltonian(h9, 9, [1,1,1],1)
    #h11 = recursive_hamiltonian(h10, 10, [1,1,1],1)

    
    #print(recursive_hamiltonian(15,[1,1,1],1))

    #hset = connected_chain_hamiltonian(4, [-1,0,0], 1)
    n_sizes = [4]
    y_values = []
    #J = [0,0,-1]
    J = [1,1,0]
    for N in n_sizes:
        hset = recursive_hamiltonian(N, J, 0)
        evs = scipy.sparse.linalg.eigsh(hset)[0]

        #if N == 2:
        #    print("TEST")
        #    #print(hset@np.array([0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0]))
        print(scipy.sparse.linalg.eigsh(hset,k=2**N))
        print(scipy.sparse.linalg.eigsh(hset,k=2**N)[1][:,8])
        plt.hlines(evs,-1,1)
        plt.show()
        evs.sort()
        y_values.append(evs[1]-evs[0])

    import scipy
    #def f(x, a,b,c):
    #    return a*np.exp(b*x)+c
    #popt, pcov = scipy.optimize.curve_fit(f, n_sizes, y_values, p0 = [1,-1,1])
    plt.scatter(n_sizes,y_values)
    #plt.plot(n_sizes, f(n_sizes, *popt), color="red", linestyle = "dashed")
    #print(f"Above x-axis by (c in f = ae^bx + c): {popt[2]}")
    plt.xlabel(r"Size of system $N$")
    plt.ylabel(r"Energy gap $\Delta E = E_1 - E_0$")
    plt.show()
    #print(scipy.sparse.linalg.eigsh(hset)[1])
    #print(scipy.sparse.linalg.eigsh(hset)[1][:,0])
    #print(trotterization_exact.minimal_eigenvector(hset))

    #times = []
    #for size in range(1,10):
    #    tstart = time.time()
    #    hset = hamiltonian_of_set(size, [-1,0,0],1)
    #    import trotterization_exact
    #    print(f"hamiltonian: {hset}")
    #    print(scipy.sparse.linalg.eigsh(hset))
    #    print(trotterization_exact.minimal_eigenvector(hset))
    #    tend = time.time()
    #    times.append(tend-tstart)

        #input()

    #plt.plot(range(1,10), times)
    #plt.show()

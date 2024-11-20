# Code for implementing Spin Fusion method on classical computer.
# Exponentiates intermediate hamiltonian.
# Does not use sparse matricies, thus slow and depricatd.

import numpy as np
import scipy
from matplotlib import pyplot as plt
import spin_chain_improved

# recursive_hamiltonian takes: (size, [Jx, Jy, Jz], h)
hset2 = spin_chain_improved.recursive_hamiltonian(2,[1,0,0],3)
hset2p = np.kron(hset2, np.identity(4)) + np.kron(np.identity(4), hset2)
hset4 = spin_chain_improved.recursive_hamiltonian(4,[1,0,0],3)



# Negative J for antiferromagnetic case
# List of dimensions of initial hamiltonians to couple together
def generate_hamiltonians(initial_dim, J, uniform_h):
    full_initial = (1j)*np.zeros((2**np.sum(initial_dim), 2**np.sum(initial_dim)))
    for i in range(len(initial_dim)):
        curr_dim = initial_dim[i]
        initial_term = spin_chain_improved.recursive_hamiltonian(curr_dim, J,uniform_h)
        initial_term = np.kron(np.identity(2**(int(np.sum(initial_dim[:i])))), initial_term)
        initial_term = np.kron(initial_term, np.identity(2**int(np.sum(initial_dim[i+1:]))))

        full_initial += initial_term

    full_final = spin_chain_improved.recursive_hamiltonian(np.sum(initial_dim), J,uniform_h)

    return full_initial, full_final

def minimal_eigenvector(matrix):
    w, v = scipy.linalg.eigh(matrix)
    return v[:,0]

mis = []
mfs = []
h_values = []
for i in range(0,400,20):
    h_value = i*0.01
    h_values.append(h_value)
    ghams = generate_hamiltonians([2,2], [-1,0,0], h_value)
    print(minimal_eigenvector(ghams[0]))
    mis.append(ghams[0])
    mfs.append(ghams[1])


# mis: initial hamiltonians
# mfs: final hamiltonians
# rss: runtime steps
#mis = [I1, I2, I3, I4, I5]
#mfs = [F1, F2, F3, F4, F5]
rss = [int(i) for i in np.linspace(10,10,1200)]
#mis = [hset2]*100
#mfs = [hset4]*100
#rss = [int(i) for i in np.linspace(1,100,100)]

r_set = []
overlaps_squared = []


if True:
    for i in range(len(mfs)):
        runtime_steps = rss[i] 
        mf = mfs[i]
        mi = mis[i]
        SIZE = len(mi)
        ground_state = minimal_eigenvector(mi)
        print(f"ground state: {ground_state}")
        overlap_squared = np.matmul(minimal_eigenvector(mf).conj(),ground_state)**2
        overlaps_squared.append(overlap_squared)
        print(f"Overlap of initial ground state and final ground state: {overlap_squared}")

    plt.scatter(h_values, overlaps_squared)
    plt.show()

for i in range(len(mfs)):
    runtime_steps = rss[i] 
    mf = mfs[i]
    mi = mis[i]
    SIZE = len(mi)

    ground_state = minimal_eigenvector(mi)
    #overlap_squared =np.matmul(minimal_eigenvector(mf).conj(),ground_state)**2
    #overlaps_squared.append(overlap_squared)
    #print(f"Overlap of initial ground state and final ground state: {overlap_squared}")

    current_state = ground_state

    s_record = []
    ev_record = []

    def interpolated_matrix(matrix_i, matrix_f, s):
        return (1-s)*matrix_i + s*matrix_f

    # adiabatic evolution
    for t in range(0,runtime_steps):
        s = 1/runtime_steps * t
        
        #mc = sparse.csr_matrix(interpolated_matrix(mi,mf,s))
        mc = interpolated_matrix(mi,mf,s)

        #me = sparse.linalg.expm((-1j)*mc)
        me = scipy.linalg.expm((-1j)*mc)
        #print(me)
        
        #print(current_state)
        current_state = np.matmul(me,current_state)
        #current_state = me@current_state

        ev_record.append(scipy.linalg.eigh(mc)[0])
        s_record.append(s)

    ev_record = np.array(ev_record)
    res = []
    wff, vrff = scipy.linalg.eigh(mf)
    #print(wff)
    for j in range(SIZE):
        # assuming the complete set of eigenstates in the final hamiltonian
        # can be represented by (1 0 0 0 0) (0 1 0 0 0) ...
        # calculate projection |<eigenstate_n | final state>|^2
        # |<ground state | final state>|^2 ~ 1 indicates success
        prepare = vrff[:,j]
        #print(prepare)
        res.append(abs(np.matmul(prepare.conj(), current_state))**2)
        #print(res)
        #print(f"GROUND STATE OVERLAP: {abs(np.matmul(vrff[:, np.argmin(wff)].conj(), current_state))**2}")

    overlaps = res/np.sum(res)
    r_set.append(abs(overlaps[0]-1))
    print(f"Overlaps: {overlaps}")
    #print(f"Eigenvalues: {[float(i)for i in wff]}")
    print(f"Ground State Overlap: {abs(np.matmul(minimal_eigenvector(mf).conj(),current_state))**2}")

    #plt.scatter([float(i) for i in wff], overlaps)
    #plt.show()
       
    if i == len(mfs)-1:
        for j in range(SIZE):
            plt.plot(s_record, ev_record[:,j], color="black")
        plt.show()

fig = plt.figure()

ax = fig.add_subplot(2, 1, 1)

ax.scatter(h_values, r_set)

ax.set_yscale('log')

plt.show()


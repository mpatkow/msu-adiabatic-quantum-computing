import numpy as np
from numpy import linalg
import scipy
from matplotlib import pyplot as plt
import spin_chain_improved

# Possible hamiltonians
pauli_x = np.array([[0,1],[1,0]])
pauli_z = np.array([[1,0],[0,-1]])
simple_nondegen = np.array([[3,0,0],[0,2,0],[0,0,1]])
random_m_NOTHERMITIAN = np.array(np.random.rand(3,3))
random_m = random_m_NOTHERMITIAN + random_m_NOTHERMITIAN.conj().T 
simple_degen = np.array([[3,0,0],[0,1,0],[0,0,1]])
random_10_n = np.array(np.random.rand(10,10))
random_10 = random_10_n + random_10_n.conj().T
nondegen_10 = np.zeros((10,10))
for i in range(10):
    nondegen_10[i][i] = 10-(i)
degen_10 = np.matrix.copy(nondegen_10)
for i in range(7,10):
    degen_10[i][i] = -1 
random_10_e = np.zeros((11,11))
nondegen_10_e = np.zeros((11,11))
random_10_e[:10, :10] = random_10[:,:]
nondegen_10_e[:10, :10] = nondegen_10[:,:]
random_10_e[10,10]=-1
nondegen_10_e[10,10]=1
random_10_e[10] = random_10_e[10] + random_10_e[9] + random_10_e[8] + random_10_e[7]

# hamiltonian of set takes: (size, [Jx, Jy, Jz], h)
hset2 = spin_chain.recursive_hamiltonian(4,[1,0,0.26],3)
hset2p = np.kron(hset2, np.identity(16)) + np.kron(np.identity(16), hset2)
print("!!")
hset4 = spin_chain.hamiltonian_of_set(16,[1,0,0.26],3)
print("!!")

# mis: initial hamiltonians
# mfs: final hamiltonians
# rss: runtime steps
mis = [hset2p]*100
mfs = [hset4]*100
rss = [int(i) for i in np.linspace(1,2000,1000)]
#mis = [hset2]*100
#mfs = [hset4]*100
#rss = [int(i) for i in np.linspace(1,100,100)]

r_set = []

def minimal_eigenvector(matrix):
    w, v = scipy.linalg.eigh(matrix)
    return v[:,0]

for i in range(len(mfs)):
    runtime_steps = rss[i] 
    mf = mfs[i]
    mi = mis[i]
    SIZE = len(mi)

    ground_state = minimal_eigenvector(mi)


    current_state = ground_state

    s_record = []
    ev_record = []

    def interpolated_matrix(matrix_i, matrix_f, s):
        return (1-s)*matrix_i + s*matrix_f

    # adiabatic evolution
    for t in range(0,runtime_steps):
        s = 1/runtime_steps * t
        
        mc = interpolated_matrix(mi,mf,s)

        me = scipy.linalg.expm((-1j)*mc)

        current_state = np.matmul(me,current_state)

        ev_record.append(scipy.linalg.eigh(mc)[0])
        s_record.append(s)

    ev_record = np.array(ev_record)
    res = []
    wff, vrff = scipy.linalg.eigh(mf)
    print(wff)
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

    overlaps = res/sum(res)
    r_set.append(abs(overlaps[0]-1))
    print(f"Overlaps: {overlaps}")
    print(f"Eigenvalues: {[float(i)for i in wff]}")
    print(f"Ground State Overlap: {abs(np.matmul(minimal_eigenvector(mf).conj(),current_state))**2}")

    #plt.scatter([float(i) for i in wff], overlaps)
    #plt.show()
       
    if i == len(mfs)-1:
        for j in range(SIZE):
            plt.plot(s_record, ev_record[:,j], color="black")
        plt.show()

fig = plt.figure()

ax = fig.add_subplot(2, 1, 1)

ax.scatter(np.linspace(1,len(r_set), len(r_set)), r_set)

ax.set_yscale('log')

plt.show()

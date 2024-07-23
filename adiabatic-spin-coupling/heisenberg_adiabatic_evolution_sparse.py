import numpy as np
import scipy
from matplotlib import pyplot as plt
import spin_chain_improved
import time


# Generate a set of hamiltonians that, at first, have no coupling
# and only calculate internal spin-spin couplings. The final hamiltonian
# will have dimension sum(initial_dim) and will include all spin-spin terms.
#
# initial_dim: list of dimensions of initial hamiltonians to couple together
def generate_initial_and_final_hamiltonians(initial_dim, J, uniform_h):
    dim_sum = sum(initial_dim)
    full_initial = scipy.sparse.csr_matrix((2**dim_sum, 2**dim_sum), dtype=complex)
    for i in range(len(initial_dim)):
        curr_dim = initial_dim[i]
        initial_term = spin_chain_improved.recursive_hamiltonian_sparse(curr_dim, J,uniform_h)
        left_identity_size = 2**(int(np.sum(initial_dim[:i])))
        right_identity_size = 2**(int(np.sum(initial_dim[i+1:])))
        initial_term = scipy.sparse.kron(scipy.sparse.identity(left_identity_size), initial_term)
        initial_term = scipy.sparse.kron(initial_term, scipy.sparse.identity(right_identity_size))
        full_initial += initial_term

    full_final = spin_chain_improved.recursive_hamiltonian_sparse(np.sum(initial_dim), J,uniform_h)

    return full_initial, full_final

def minimal_eigenvector(matrix):
    eigen_result = scipy.sparse.linalg.eigsh(matrix)
    return eigen_result[1][:,np.argmin(eigen_result[0])]

def overlap_squared(u,v):
    return abs(u.conj() @ v)**2


if __name__ == "__main__":
    # mis: initial hamiltonians
    # mfs: final hamiltonians
    # rss: runtime steps for corresponding mis, mfs pair
    mis = []
    mfs = []
    rss = [int(i) for i in np.linspace(50,50,10)]

    h_values = np.linspace(0,500,10)*0.01 # We will vary the magnetic field h in our trials
    #h_values = np.linspace(0,0,40)
    J = [-1,-1,0] # Negative J for antiferromagnetic case

    for h_value in h_values:
        generated_hamiltonians = generate_initial_and_final_hamiltonians([3,3], J, h_value)
        mis.append(generated_hamiltonians[0])
        mfs.append(generated_hamiltonians[1])

    r_set = []
    overlaps_squared = []


    # Calculate overlap of ground state of initial hamiltonian and final hamiltonian
    for i in range(len(mfs)):
        runtime_steps = rss[i] 
        mf = mfs[i]
        mi = mis[i]
        ground_state = minimal_eigenvector(mi)
        os = overlap_squared(minimal_eigenvector(mf),ground_state)
        overlaps_squared.append(os)


    for i in range(len(mfs)):
        runtime_steps = rss[i] 
        mf = mfs[i]
        mi = mis[i]
        #SIZE = len(mi)

        ground_state = minimal_eigenvector(mi)
        #overlap_squared =np.matmul(minimal_eigenvector(mf).conj(),ground_state)**2
        #overlaps_squared.append(overlap_squared)
        #print(f"Overlap of initial ground state and final ground state: {overlap_squared}")

        mi = scipy.sparse.csc_matrix(mi)
        mf = scipy.sparse.csc_matrix(mf)

        current_state = scipy.sparse.csc_matrix(ground_state).transpose()

        s_record = []
        ev_record = []

        def interpolated_matrix_linear(matrix_i, matrix_f, s):
            return (1-s)*matrix_i + s*matrix_f

        def interpolated_matrix_special(matrix_i, matrix_f, s):
            return (1-np.sin(np.pi/2*s)**2)*matrix_i + (np.sin(np.pi/2*s)**2)*matrix_f

        # adiabatic evolution
        for t in range(0,runtime_steps):
            s = 1/runtime_steps * t
            
            mc = interpolated_matrix_special(mi,mf,s)

            #print(current_state)
            t1 = time.time()
            current_state = scipy.sparse.linalg.expm_multiply((-1j)*mc,current_state)
            t2 = time.time()

            print(t2-t1)
            #current_state = me@current_state
            #eigen_result = scipy.sparse.linalg.eigsh(mc)
            #print(eigen_result[1][:,np.argmin(eigen_result[0])])
            #ev_record.append(eigen_result[0][np.argmin(eigen_result[0])])
            #ev_record.append(scipy.linalg.eigh(mc)[0])
            #s_record.append(s)

        print("Done")
        ev_record = np.array(ev_record)
        res = []
        wff, vrff = scipy.sparse.linalg.eigsh(mf)
        #print(wff)
        for j in range(len(wff)):
            # assuming the complete set of eigenstates in the final hamiltonian
            # can be represented by (1 0 0 0 0) (0 1 0 0 0) ...
            # calculate projection |<eigenstate_n | final state>|^2
            # |<ground state | final state>|^2 ~ 1 indicates success
            prepare = vrff[:,j]
            #print(prepare)
            res.append(abs(prepare.conj()@current_state)**2)
            #print(res)
            g_state_overlap = overlap_squared(minimal_eigenvector(mf), current_state)
            print(f"GROUND STATE OVERLAP: {g_state_overlap}")

        overlaps = res/np.sum(res)
        #r_set.append(abs(overlaps[0]-1))
        r_set.append(abs(g_state_overlap-1))
        print(f"Overlaps: {overlaps}")
        #print(f"Eigenvalues: {[float(i)for i in wff]}")
        #print(f"Ground State Overlap: {abs(np.matmul(minimal_eigenvector(mf).conj(),current_state))**2}")

        #plt.scatter([float(i) for i in wff], overlaps)
        #plt.show()
        
        #if i == len(mfs)-1:
        #    for j in range(SIZE):
        #        plt.plot(s_record, ev_record[:,j], color="black")
        #    plt.show()

    figure, axis = plt.subplots(2, 2) 

    #axis[0,0].plot(h_values, overlaps_squared) 
    axis[0,0].plot(h_values, overlaps_squared) 
    axis[0,0].set_title(r"Overlaps of ground states of $H_i$ and $H_f$") 
    axis[0,0].set_xlabel(r"Magnetic field $h$") 
    axis[0,0].set_ylabel(r"$|\langle 0 |_{H_i}  |0 \rangle _{H_f}|^2$") 

    #axis[0,1].plot(h_values, r_set) 
    axis[0,1].plot(h_values, r_set) 
    axis[0,1].set_title(r"Overlap of adiabatic result and ground state of $H_f$") 
    axis[0,1].set_xlabel(r"Magnetic field $h$") 
    axis[0,1].set_ylabel(r"$1-|\langle \psi |  |0 \rangle _{H_f}|^2$") 
    axis[0,1].set_yscale('log')

    plt.show()

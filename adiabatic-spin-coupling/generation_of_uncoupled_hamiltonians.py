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
        initial_term = spin_chain_improved.connected_chain_hamiltonian(curr_dim, J,uniform_h)
        left_identity_size = 2**(int(np.sum(initial_dim[:i])))
        right_identity_size = 2**(int(np.sum(initial_dim[i+1:])))
        initial_term = scipy.sparse.kron(scipy.sparse.identity(left_identity_size), initial_term)
        initial_term = scipy.sparse.kron(initial_term, scipy.sparse.identity(right_identity_size))
        full_initial += initial_term

    full_final = spin_chain_improved.connected_chain_hamiltonian(np.sum(initial_dim), J,uniform_h)

    return full_initial, full_final

def minimal_eigenvector(matrix):
    eigen_result = scipy.sparse.linalg.eigsh(matrix)
    return eigen_result[1][:,np.argmin(eigen_result[0])]

def overlap_squared(u,v):
    return abs(u.conj() @ v)**2


def interpolated_matrix_linear(matrix_i, matrix_f, s):
    return (1-s)*matrix_i + s*matrix_f

if __name__ == "__main__":
    import scipy
    from qiskit.quantum_info import SparsePauliOp, Operator
    J = [-1,0,0]
    h =0.5 
    gened_hams = generate_initial_and_final_hamiltonians([2,2,2], J, h)
    #gened_hams = generate_initial_and_final_hamiltonians([2,2,2,2], [0,0,-1], 5)
    #gened_hams = spin_chain_improved.connected_chain_hamiltonian(9, [-1,0,0], 4)
    
    h0 = gened_hams[0].toarray()
    #h0 = gened_hams
    h1 = gened_hams[1]
    #h1 = gened_hams

    eigr0 = scipy.linalg.eig(h0)
    eigr05 = scipy.linalg.eig(interpolated_matrix_linear(h0,h1,0.5)) 
    eigr1 = scipy.linalg.eig(h1)
    evs0 = [n.real for n in eigr0[0]]
    evs1 = [n.real for n in eigr1[0]]
    evs05 = [n.real for n in eigr05[0]]
    print(f"Eigenvalues of uncoupled: {evs0}")
    print(f"Eigenvalues of 0.5: {evs05}")
    print(f"Eigenvalues of coupled: {evs1}")

    print(eigr1)

    """
    print(np.round(eigr1[1][:,0], decimals = 5))
    print(np.round(eigr1[1][:,4], decimals = 5))
    print(np.round(eigr1[1][:,5], decimals = 5))
    print(np.round(eigr1[1][:,6], decimals = 5))
    print(np.round(eigr1[1][:,7], decimals = 5))
    print(np.round(eigr1[1][:,9], decimals = 5))
    print(np.round(eigr1[1][:,10], decimals = 5))
    """
    """
    for i in range(len(eigr1[1][:,7])):
        c = eigr1[1][:,3][i]/eigr1[1][:,7][i]
        print(c)
    """
    #print(evs1)

    plt.subplot(2,2,1)
    plt.hlines(evs0,-1,1,color="red",label="unconnected")
    plt.legend()
    plt.subplot(2,2,2)
    plt.hlines(evs05,-1,1,color="blue",label="0.5 connected")
    plt.legend()
    plt.subplot(2,2,3)
    plt.hlines(evs1,-1,1,color="green",label="connected", linestyle="dashed")
    plt.legend()
    plt.subplot(2,2,4)
    size = len(evs1) 
    yv = sorted(list(set([round(i,8) for i in sorted(evs1)[0:size+1]])))
    print(yv)
    print(f"(E1-) - (E0): {yv[1]-yv[0]}")
    print(f"2(h-|J|): {2*(h-np.abs(J[0]))}")
    plt.scatter(np.linspace(1,len(yv), len(yv)), yv)
    plt.show()


    #print(SparsePauliOp.from_operator(Operator(h0)))
    #print(SparsePauliOp.from_operator(Operator(h1)))

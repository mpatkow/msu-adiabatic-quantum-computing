# WIP code implementing AQC simulation modified to easily verify if tenpy code is working.

import numpy as np
import scipy
from matplotlib import pyplot as plt
import spin_chain_improved
import time
from tqdm import tqdm

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

def run(SHAPE, TOTAL_TIME, TROTTER_STEPS, D_TIME, CONSTANT_H, J, SHOW_PLOT, INTERPOLATION_TYPE):
    print(SHAPE, TOTAL_TIME, TROTTER_STEPS, D_TIME, CONSTANT_H, J, SHOW_PLOT, INTERPOLATION_TYPE)

    if TROTTER_STEPS != 0:
        D_TIME = TOTAL_TIME/TROTTER_STEPS
    else:
        D_TIME = 1

    if INTERPOLATION_TYPE not in ["linear", "special"]:
        print("Not working, select proper interpolation type")
        return -1
    generated_hamiltonians_mag = generate_initial_and_final_hamiltonians(SHAPE, [0,0,0], CONSTANT_H)
    generated_hamiltonians_int = generate_initial_and_final_hamiltonians(SHAPE, J, 0)
    generated_hamiltonians_full = generate_initial_and_final_hamiltonians(SHAPE, J, CONSTANT_H)
    mi_mag = generated_hamiltonians_mag[0]
    mf_mag = generated_hamiltonians_mag[1]
    mi_int = generated_hamiltonians_int[0]
    mf_int = generated_hamiltonians_int[1]

    mi_mag = scipy.sparse.csc_matrix(mi_mag)
    mf_mag = scipy.sparse.csc_matrix(mf_mag)
    mi_int = scipy.sparse.csc_matrix(mi_int)
    mf_int = scipy.sparse.csc_matrix(mf_int)

    current_state_i = [0]*2**(sum(SHAPE))
    current_state_i[0] = 1
    current_state_i = minimal_eigenvector(generated_hamiltonians_full[0])
    current_state_i = np.array(current_state_i)/np.sum(current_state_i.conj() @ current_state_i)
    current_state = scipy.sparse.csc_matrix(current_state_i).transpose()

    s_record = []
    ev_record = []

    def interpolated_matrix_linear(matrix_i, matrix_f, s):
        return (1-s)*matrix_i + s*matrix_f

    def interpolated_matrix_special(matrix_i, matrix_f, s):
        return (1-np.sin(np.pi/2*s)**2)*matrix_i + (np.sin(np.pi/2*s)**2)*matrix_f

    # adiabatic evolution
    for s in np.linspace(0,1,TROTTER_STEPS):
        if INTERPOLATION_TYPE == "linear":
            mc_mag = interpolated_matrix_linear(mi_mag,mf_mag,s)
            mc_int = interpolated_matrix_linear(mi_int,mf_int,s)
        elif INTERPOLATION_TYPE == "special":
            mc_mag = interpolated_matrix_special(mi_mag,mf_mag,s)
            mc_int = interpolated_matrix_special(mi_int,mf_int,s)

        current_state = scipy.sparse.linalg.expm_multiply((-1j)*(mc_int+mc_mag)*D_TIME,current_state)
        #current_state = scipy.sparse.linalg.expm_multiply((-1j)*mc_mag*D_TIME,current_state)

        ev_record.append(scipy.sparse.linalg.eigsh(mc_mag+mc_int)[0])
        s_record.append(s)

    current_state = current_state.transpose().toarray()[0]
    mf = spin_chain_improved.connected_chain_hamiltonian(sum(SHAPE), J, CONSTANT_H) 

    e_exp = float(np.inner(np.conj(current_state).T, np.matmul(mf,current_state)).real)
    print(f"e_exp {e_exp}")

    # TODO, SHOULD CHANGE INDEX TO 1 TO BE SECOND SMALLEST EIGENVECTOR!!
    the_eigsh_of_final = scipy.sparse.linalg.eigsh(generated_hamiltonians_full[1])
    eigenvalues_sorted = sorted(the_eigsh_of_final[0])
    second_smallest_eigenvalue = eigenvalues_sorted[3]
    print(eigenvalues_sorted)
    #print(second_smallest_eigenvalue)
    second_smallest_eigenvalue_index = np.where(the_eigsh_of_final[0] == second_smallest_eigenvalue)[0][0]
    #print(second_smallest_eigenvalue_index)

    #print(the_eigsh_of_final[0][second_smallest_eigenvalue_index])

    second_smallest_eigenvector = the_eigsh_of_final[1][:,second_smallest_eigenvalue_index]
    #print(f"by new code: {second_smallest_eigenvalue_index}")
    ##print(second_smallest_eigenvector)
    #print(minimal_eigenvector(mf))
    #print("break")

    if len(ev_record) < 1:
        ev_record = [[-1,-1,-1]]
    ground_state_energy = min(ev_record[-1])
    print(f"Ground State Energy (E_0): {ground_state_energy}")
    energies = sorted(ev_record[-1])

    #print(s_record)
    #print(ev_record)
    if SHOW_PLOT:
        for i in range(len(s_record)):
            for j in range(len(ev_record[i])):
                #print(s_record[i])
                #print(ev_record[i])
                plt.scatter(s_record[i], ev_record[i][j])
        plt.show()

    print(minimal_eigenvector(mf))
    g_state_overlap = overlap_squared(minimal_eigenvector(mf), current_state)
    e1_state_overlap = overlap_squared(second_smallest_eigenvector, current_state)

    return e_exp, ground_state_energy, g_state_overlap, energies, e1_state_overlap

if __name__ == "__main__":
    TOTAL_TIME = 1
    TROTTER_STEPS = 10
    D_TIME = TOTAL_TIME/TROTTER_STEPS
    CONSTANT_H = 1
    J = np.array([-1,0,0])
    SHOW_PLOT = False
    e_values = []
    e0_values = []
    go_values = []
    e1_values = []
    LIMITING_DIMENSION = 1
    BEGINNING_CUTOFF = 1
    shape_values = [[2,2],[2,2,2],[2,2,2,2],[2,2,2,2,2],[2,2,2,2,2,2]][:LIMITING_DIMENSION]
    #shape_values = [[4,4],[4,4,4]]
    #shape_values = [[5,5],[5,5,5]]
    #shape_values = [[3,3],[3,3,3],[3,3,3,3],[3,3,3,3,3],[3,3,3,3,3,3]][:LIMITING_DIMENSION]
    #shape_values = [[8,8]]
    energies_values = []
    #h_values = np.linspace(0,4,40) 
    h_values = np.linspace(1,1,3) 
    j_coeffs = np.linspace(0,3,20)
    total_runtimes = np.linspace(0,10,30)
    for SHAPE in tqdm(shape_values):
        e_values_for_a_run = []
        e0_values_for_a_run = []
        go_values_for_a_run = []
        energies_values_for_a_run = []
        e1_values_for_a_run = []
        for TOTAL_TIME in total_runtimes:
            run_res = run(SHAPE, TOTAL_TIME, TROTTER_STEPS, D_TIME, CONSTANT_H, J, SHOW_PLOT, "linear")
            e_values_for_a_run.append(run_res[0])
            e0_values_for_a_run.append(run_res[1])
            go_values_for_a_run.append(run_res[2])
            energies_values_for_a_run.append(run_res[3][1])
            e1_values_for_a_run.append(run_res[4])
        e_values.append(e_values_for_a_run)
        e0_values.append(e0_values_for_a_run)
        go_values.append(go_values_for_a_run)
        energies_values.append(energies_values_for_a_run)
        e1_values.append(e1_values_for_a_run)

    trotter_steps_arr = np.linspace(0,len(e_values[0]), len(e_values[0]))

    # Plotting the Energy Expectation Value
    labels = ["2-2", "2-2-2", "2-2-2-2", "2-2-2-2-2", "2-2-2-2-2-2"][:LIMITING_DIMENSION]
    labels_ground_state = ["2-2 ground", "2-2-2 ground", "2-2-2-2 ground", "2-2-2-2-2 ground", "2-2-2-2-2-2 ground"][:LIMITING_DIMENSION]
    labels_second_state = ["2-2 second", "2-2-2 second", "2-2-2-2 second", "2-2-2-2-2 second", "2-2-2-2-2-2 second"][:LIMITING_DIMENSION]
    plt.subplot(2, 3, 1)
    colors = ['blue', 'orange', 'green', 'red', 'purple', "brown", "pink", "gray", "olive", "cyan"]
    for i in range(len(e_values)):
        color = colors[i]
        # other option is to use trotter_steps_arr for x axis values
        plt.scatter(total_runtimes[BEGINNING_CUTOFF:], e_values[i][BEGINNING_CUTOFF:], label = labels[i], color = color)
        plt.scatter(total_runtimes[BEGINNING_CUTOFF:], e0_values[i][BEGINNING_CUTOFF:],  color = color, linewidth=0.75, linestyle="dashed")
        print(np.subtract(energies_values[i],e0_values[i]))
        plt.scatter(total_runtimes[BEGINNING_CUTOFF:], energies_values[i][BEGINNING_CUTOFF:],  color = color, linewidth=0.75, linestyle="dashed")
        #plt.plot(np.linspace(0,len(e_values[0]), len(e_values[0])), e0_values[i], label = labels_ground_state[i], color = color, linewidth=0.75, linestyle="dashed")
        #plt.plot(np.linspace(0,len(e_values[0]), len(e_values[0])), energies_values[i], label = labels_second_state[i], color = color, linewidth=0.75, linestyle="dashed")
    plt.xlabel(r"Trotter Steps")
    #plt.xlabel(r"Magnetic field $h$")
    #plt.xlabel(r"Coupling coefficient $j$")
    #plt.xlabel(r"Total Runtime $T$")
    plt.ylabel(r"Energy Expectation Value $\langle \phi | \hat H | \phi \rangle$")
    plt.legend()

    # Plotting the Overlap
    plt.subplot(2,3,2)
    for i in range(len(e_values)):
        color = colors[i]
        yvalues = go_values[i]
        #yvalues = np.divide(np.ones(len(yvalues)), yvalues)
        plt.plot(total_runtimes[BEGINNING_CUTOFF:], yvalues[BEGINNING_CUTOFF:], label = labels[i], color = color)
        #plt.plot(h_values[BEGINNING_CUTOFF:], e1_values[i][BEGINNING_CUTOFF:], label = labels[i], color = color)
    plt.xlabel(r"Trotter Steps")
    #plt.xlabel(r"Magnetic field $h$")
    #plt.xlabel(r"Coupling coefficient $j$")
    #plt.xlabel(r"Total Runtime $T$")
    plt.ylabel(r"Fidelity (Ground State Overlap) $|\langle \phi | \psi_0 \rangle|^2$")
    plt.legend()

    plt.subplot(2,3,3)
    for i in range(len(e_values)):
        color = colors[i]
        yvalues = []
        for j in range(len(e_values[i])):
            evalue = e_values[i][j]
            e0value = e0_values[i][j]
            yvalues.append(np.abs((evalue-e0value)/e0value))
        
        #yvalues = np.divide(np.ones(len(yvalues)), yvalues)

        plt.plot(total_runtimes[BEGINNING_CUTOFF:], yvalues[BEGINNING_CUTOFF:], color = colors[i], label = labels[i])
    plt.xlabel(r"Trotter Steps")
    #plt.xlabel(r"Magnetic field $h$")
    #plt.xlabel(r"Coupling coefficient $j$")
    #plt.xlabel(r"Total Runtime $T$")
    plt.ylabel(r"Absolute Relative Energy Error $|r| = \left|(\langle E \rangle - E_0)/ E_0\right|$")
    plt.legend()

    plt.subplot(2,3,4)
    for i in range(len(e_values)):
        plt.scatter(np.subtract(energies_values[i], e0_values[i])[BEGINNING_CUTOFF:], go_values[i][BEGINNING_CUTOFF:], color=colors[i], label = labels[i])

    plt.xlabel(r"Minimum Energy Difference $\Delta E = E_1 - E_0 $")
    plt.ylabel(r"Fidelity (Ground State Overlap) $|\langle \phi | \psi_0 \rangle|^2$")
    plt.legend()

    plt.subplot(2,3,5)
    #for i in range(len(e_values)):
    #    x_values_to_plot = np.subtract(energies_values[i], e0_values[i])
    #    plt.scatter(x_values_to_plot, 2*(h_values-np.abs(J[0])), color = colors[i], label = labels[i])
    #    smallest_v = x_values_to_plot[0]
    #    largest_v = x_values_to_plot[-1]
    #plt.plot(np.linspace(smallest_v, largest_v, 20), np.linspace(smallest_v, largest_v, 20), label = "x=y", color="black")
    plt.xlabel(r"Energy gap: $E_1^- - E_0$")
    plt.ylabel(r"Approximation: $2(h-|J|)$")
    plt.legend()


    # Plotting the Overlap
    plt.subplot(2,3,6)
    for i in range(len(e_values)):
        color = colors[i]
        yvalues = go_values[i]
        #yvalues = np.divide(np.ones(len(yvalues)), yvalues)
        #plt.plot(h_values[BEGINNING_CUTOFF:], yvalues[BEGINNING_CUTOFF:], label = labels[i], color = color)
        plt.plot(total_runtimes[BEGINNING_CUTOFF:], e1_values[i][BEGINNING_CUTOFF:], label = labels[i], color = color)
    plt.xlabel(r"Trotter Steps")
    #plt.xlabel(r"Magnetic field $h$")
    #plt.xlabel(r"Coupling coefficient $j$")
    #plt.xlabel(r"Total Runtime $T$")
    plt.ylabel(r"First Excited Fidelity $|\langle \phi | \psi_1 \rangle|^2$")
    plt.legend()



    plt.show()

"""
if __name__ == "__main__":
    # mis: initial hamiltonians
    # mfs: final hamiltonians
    # rss: runtime steps for corresponding mis, mfs pair
    mis = []
    mfs = []
    rss = [int(i) for i in np.linspace(50,50,10)]

    h_values = np.linspace(0,0,10)*0.01 # We will vary the magnetic field h in our trials
    #h_values = np.linspace(0,0,40)
    J = [0,0,-1] # Negative J for antiferromagnetic case

    for h_value in h_values:
        generated_hamiltonians = generate_initial_and_final_hamiltonians([2,2], J, h_value)
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

        #ground_state = minimal_eigenvector(mi)
        states = [np.array(s) for s in [[1,0],[1,0],[1,0],[1,0]]]
        ground_state = np.kron(states[0],states[1])
        for state in states[2:]:
            ground_state = np.kron(ground_state, state)
        print(ground_state)
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
            
            mc = interpolated_matrix_linear(mi,mf,s)

            #print(current_state)
            current_state = scipy.sparse.linalg.expm_multiply((-1j)*mc,current_state)

            print(f"Current State: {current_state.toarray()}")
            input()

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
    """

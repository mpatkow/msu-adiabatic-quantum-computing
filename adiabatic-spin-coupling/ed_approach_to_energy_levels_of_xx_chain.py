import numpy as np

def generate_model(L):
    ham = np.zeros((L,L))

    # populate upper diagonal 
    for i in range(L-1):
        ham[i][i+1] = 1

    # populate lower diagonal 
    for i in range(L-1):
        ham[i+1][i] = 1

    ham /= 2 # factors due to representation of hamiltonian in terms of S+ and S- operators
    return ham

def get_energies_half_filled(model):
    eigenvalues = np.sort(np.linalg.eigvals(model))
    L = len(model)
    ground_state_E = np.sum(eigenvalues[:L//2])
    first_excited_E = np.sum(eigenvalues[:L//2-1]) + eigenvalues[L//2]

    return ground_state_E, first_excited_E
    



def main():
    import matplotlib.pyplot as plt
    sizes = [4,8,12,16,20]# np.arange(2,100,2)
    energy_differences = []
    for size in sizes:
        hamiltonian = generate_model(size)
        E_0, E_1 = get_energies_half_filled(hamiltonian)

        energy_differences.append(E_1-E_0)
        #print(f"ground state energy: {E_0}\nfirst excited energy: {E_1}\nDelta E: {E_1-E_0}")

    print(energy_differences)

    plt.plot(sizes,energy_differences)
    plt.title(r"System size vs. Energy Gap with ED for half-filled chains")
    plt.xlabel(r"System size $L$")
    plt.ylabel(r"Energy Difference = $\Delta E = E_1 - E_0$")
    plt.show()


if __name__ == "__main__":
    main()

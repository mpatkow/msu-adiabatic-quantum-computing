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

def get_energies_quarter_filled(model):
    eigenvalues = np.sort(np.linalg.eigvals(model))
    L = len(model)
    ground_state_E = np.sum(eigenvalues[:L//4])
    first_excited_E = np.sum(eigenvalues[:L//4-1]) + eigenvalues[L//2]

    return ground_state_E, first_excited_E



def main():
    import sys
    hamiltonian = generate_model(int(sys.argv[1]))
    energies = get_energies_quarter_filled(hamiltonian)
    print(energies[0])
    print(f"gap: {energies[1] - energies[0]}")


    exit()


    import matplotlib.pyplot as plt
    sizes = [i+1 for i in range(3,200)]
    #sizes = [2**i for i in range(10)]
    energy_differences = []
    for size in sizes:
        hamiltonian = generate_model(size)
        E_0, E_1 = get_energies_half_filled(hamiltonian)
        #E_0, E_1 = get_energies_quarter_filled(hamiltonian)

        energy_differences.append(E_1-E_0)
        #print(f"ground state energy: {E_0}\nfirst excited energy: {E_1}\nDelta E: {E_1-E_0}")

    from scipy.optimize import curve_fit

    for s,ediff in zip(sizes, energy_differences):
        print(f"{s} {round(ediff,7)}")
 
    def test(x, m, b, c, e):
        return m/(x-c+0.0001)**(1/2) + 0
 
    param, param_cov = curve_fit(test, sizes, energy_differences)# p0=(2,0,1,1))
    
    ans = test(sizes,*param) 

    print(*param)
    print(test(8,*param))

    energy_differences = np.array(energy_differences)
    plt.scatter(sizes,np.divide(np.ones(len(energy_differences)),np.power(energy_differences,1)))
    plt.plot(sizes, ans)
    plt.title(r"System size vs. Energy Gap with ED for half-filled chains")
    plt.xlabel(r"System size $L$")
    plt.ylabel(r"Energy Difference = $\Delta E = E_1 - E_0$")
    plt.show()


if __name__ == "__main__":
    main()

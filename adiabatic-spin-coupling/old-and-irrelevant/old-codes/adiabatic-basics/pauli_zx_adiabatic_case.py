import numpy as np
from numpy import linalg
import scipy

pauli_x = np.array([[0,1],[1,0]])
pauli_z = np.array([[1,0],[0,-1]])

mi = pauli_x
mf = pauli_z

runtime_steps = 100000

initial_state = 1/np.sqrt(2) * np.array([[1],[-1]])
current_state = initial_state


def interpolated_matrix(matrix_i, matrix_f, s):
    return (1-s)*matrix_i + s*matrix_f

for t in range(1,runtime_steps+1):
    s = 1/runtime_steps * t
    
    mc = interpolated_matrix(mi,mf,s)

    me = scipy.linalg.expm((-1j)*mc)

    current_state = np.matmul(me,current_state)

print(current_state)

ef1 = np.array([[1,0]])
ef2 = np.array([[0,1]])

print(abs(np.matmul(ef1,current_state))**2)
print(abs(np.matmul(ef2,current_state))**2)

import numpy as np

s0 = np.array([1,0])
s1 = np.array([0,1])

def kron_of_set(to_kron):
    result = to_kron[0]
    for next_term in to_kron[1:]:
        result = np.kron(result, next_term)

    return result

print(kron_of_set([s0,s0,s0,s0]))
print(kron_of_set([s0,s0,s0,s1]))
print(kron_of_set([s0,s0,s1,s0]))
print(kron_of_set([s0,s0,s1,s1]))
print(kron_of_set([s0,s1,s0,s0]))
print(kron_of_set([s0,s1,s0,s1]))
print(kron_of_set([s0,s1,s1,s0]))
print(kron_of_set([s0,s1,s1,s1]))
print(kron_of_set([s1,s0,s0,s0]))

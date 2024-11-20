import numpy as np
import matplotlib.pyplot as plt

adiabatic_times = [6]

# sigma = 1, adiabatic = 6
rodeo_cycles = [0,1,2,3,4]
s1 = [0.9996379949942523, 0.9997215977253231, 0.9997687277150595, 0.9998390018932791, 0.9998848840953145]
s2 = [0.9996379949942523, 0.9997992006907406, 0.9998748698223938, 0.9999040147111696, 0.9999454193565755]
s3 = [0.9996379949942523, 0.9998227646216049, 0.999953855623491, 0.9999537944443703, 0.9999782052574737]
s4 = [0.9996379949942523, 0.9998300026291738, 0.9999106490250512, 0.9999801992755011, 0.9999513954636229]


def norm_log(s_values):
    return np.log10([1]*len(np.array(s_values))-np.array(s_values))

plt.plot(rodeo_cycles, np.log10(np.ones(len(s1))-np.array([1-1/2**r * (1-s1[0]) for r in rodeo_cycles])), label='predicted')

plt.plot(rodeo_cycles, norm_log(s1), label="sigma=1")
plt.plot(rodeo_cycles, norm_log(s2), label="sigma=2")
plt.plot(rodeo_cycles, norm_log(s3), label="sigma=3")
plt.plot(rodeo_cycles[:len(s4)], norm_log(s4), label="sigma=4")
plt.legend()
plt.show()


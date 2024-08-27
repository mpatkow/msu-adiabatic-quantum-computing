import numpy as np
import matplotlib.pyplot as plt
import sys

filenames = sys.argv[1:]

for filename in filenames:
    with open(filename, "r") as f_opened:
        full_dataset = f_opened.readlines()

    x_data = full_dataset[0]
    y_data = full_dataset[1]

    x_data = np.array([float(x_val) for x_val in x_data.split(",")])
    y_data = np.array([float(y_val) for y_val in y_data.split(",")])

    y_data_p = np.ones(len(y_data))-y_data

    plt.plot(x_data, y_data_p, label=filename)

plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.show()

import sys
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]

with open(filename, "r") as f:
    data_raw = f.readlines()

header = {v.rstrip("\n"):i for (i,v) in enumerate(data_raw[0].split(","))}
colors = {0.1:"red", 0.01:"blue",0.001:"green",0.0001:"violet"}
for data_line_raw in data_raw[1:]:
    data_line = [float(i) for i in data_line_raw.split(",")]
    plt.scatter(data_line[header["jz"]], data_line[header["time"]], color=colors[data_line[header["target_epsilon"]]])
plt.show()
    


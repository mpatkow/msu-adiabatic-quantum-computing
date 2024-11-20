import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "adiabatic_time_frac_data.dat"

df = pd.read_csv(filename)

filtersizes = [44]

color_labels = {2:"red",4:"blue",8:"green"}

for index,row in df.iterrows():
    if row["size"] not in filtersizes:
        continue

    x_value = float(row["adiabatic_time"])/float(row["total_time"])
    y_value = float(row["total_time"])

    plt.scatter(x_value,y_value,color=color_labels[row["sigma"]],label=row["sigma"]) #,s=shapes[row["size"]])
plt.legend()
plt.show()


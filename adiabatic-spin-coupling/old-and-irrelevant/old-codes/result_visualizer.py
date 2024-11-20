import numpy as np
import matplotlib.pyplot as plt
import sys

#data_file_name = "adiabatic_rodeo_results_2222222222222222.dat"
#data_file_name = "adiabatic_rodeo_results_1616.dat"
#data_file_name = "results/adiabatic_rodeo_results_22.dat"
data_file_name = sys.argv[1]
#data_file_name = "adiabatic_rodeo_results_44.dat"
#data_file_name = "adiabatic_rodeo_results_0p5.dat"
#data_file_name = "adiabatic_rodeo_results_3232.dat"
with open(data_file_name, "r") as f:
    data_readlines = f.readlines()

header = data_readlines[0].rstrip("\n").split(",")
header_indicies = {}
for i, header_name in enumerate(header):
    header_indicies[header_name] = i
data_formatted = []
for data_line in data_readlines[1:]:
    data_formatted.append([float(data_val) for data_val in data_line.rstrip("\n").split(",")])

sigma_split_data = {}
for data_line in data_formatted:
    current_sigma = data_line[header_indicies["sigma"]]
    if current_sigma in sigma_split_data:
        sigma_split_data[current_sigma].append(data_line)
    else:
        sigma_split_data[current_sigma] = [data_line]

sigma_values = []
x_values = []
y_values = []
z_values = []
resample_values = []
epsilon = 0.001
resample_min = 1
for sigma_value, sigma_specific_dataset in sigma_split_data.items():
    x_values.append([])
    y_values.append([])
    z_values.append([])
    resample_values.append([])
    sigma_values.append(sigma_value)
    for data_line in sigma_specific_dataset:
        if data_line[header_indicies["overlap"]] < 1-epsilon:
            continue
        if data_line[header_indicies["resamples"]] < resample_min:
            continue


        # Calculate the evolution time:
        if data_line[header_indicies["r"]] == 0:
            x_values[-1].append(data_line[header_indicies["adiabatic_time"]])
            y_values[-1].append(data_line[header_indicies["adiabatic_time"]])
        else:
            x_values[-1].append(data_line[header_indicies["adiabatic_time"]])
            try:
                y_values[-1].append(1/data_line[header_indicies["success_prob"]] * (data_line[header_indicies["r"]] * data_line[header_indicies["sigma"]] + data_line[header_indicies["adiabatic_time"]]))
            except:
                y_values[-1].append(-1)
            #x_values[-1].append(1/data_line[header_indicies["success_prob"]] * (data_line[header_indicies["r"]] * data_line[header_indicies["sigma"]] + data_line[header_indicies["adiabatic_time"]]))
        #y_values[-1].append(data_line[header_indicies["overlap"]])
        z_values[-1].append(data_line[header_indicies["overlap"]])
        resample_values[-1].append(data_line[header_indicies["resamples"]])


markers = ["o" , "v" , "s" , "P", "d" ] * 100
for i, (x_value_set, y_value_set, sigma_value, z_value_set, resample_values_set) in enumerate(zip(x_values, y_values, sigma_values, z_values, resample_values)):
    if sigma_value in [sigma_value]: #[1,1.5,2,3,5,10]:
        plt.scatter(x_value_set, y_value_set, label=str(sigma_value), c=z_value_set, marker=markers[i],cmap='viridis',s=np.array(resample_values_set)*2+10)
        plt.clim(0.65,1)

plt.xlabel(r"Adiabatic Evolution Time $T_{\mathrm{AE}}$")
plt.ylabel(r"Total Time $1/P (r\sigma + T_{\mathrm{AE}})$")
plt.plot(np.linspace(0.5,10,2),np.linspace(0.5,10,2), linestyle="dashed", color="black")
plt.title(r"Runs with overlap within $\epsilon=$" + str(epsilon))
plt.legend()
cbar = plt.colorbar()
cbar.set_label("Overlap with exact ground state")
plt.show()

for i, (x_value_set, y_value_set, sigma_value, z_value_set, resample_values_set) in enumerate(zip(x_values, y_values, sigma_values, z_values, resample_values)):
    for pair_x, pair_y, pair_z in zip(x_value_set, y_value_set, z_value_set):
        if pair_x == 0:
            plt.scatter(pair_y,pair_z)
plt.show()

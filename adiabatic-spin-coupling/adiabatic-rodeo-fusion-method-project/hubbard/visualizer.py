import os
import numpy as np
import re
import matplotlib.pyplot as plt

datadir = "data/"

adiabatic_times = []
costs = []
overlaps = []

def verifysize(size,r):
    checknext  = False
    sizev = ""
    for line_r in r:
        if "shape" in line_r:
            checknext=True
            continue
        if checknext: 
            if "]" in line_r:
                return sizev==size
            else:
                sizev+=re.search(r'\d+', line_r).group()





for datafilename in os.listdir(datadir):
    with open(datadir + datafilename, "r") as f:
        r = f.readlines()
        if "cost" in r[-2] and verifysize("3232",r):
            for line_r in r:
                if "adiabatic_time" in line_r:
                    adiabatic_times.append(float(line_r.split(" ")[-1].rstrip(",\n")))
            measured_values = [float(mesval) for mesval in r[-1].split(",")]
            costs.append(measured_values[2])
            overlaps.append(measured_values[0])

x_values = np.array([np.divide(adiabatic_times,costs)])
x_values[np.isnan(x_values)] = 0
y_values_unnorm = np.array([costs])
y_values = y_values_unnorm
z_values = np.abs(np.log10(np.ones(len(overlaps))-overlaps))

minlim = 2
maxlim = 100
plt.scatter(x_values, y_values, c=z_values*(z_values>minlim)*(z_values<maxlim))
plt.xlabel(r"adiabatic time ratio: $\frac{T_\mathrm{RA}}{T}$")
plt.ylabel(r"total time: $T$")
#plt.yticks(np.linspace(0,1,ytickslen), np.linspace(0,int(np.max(y_values_unnorm[0]))+1, ytickslen)) #, rotation='vertical')
plt.colorbar()
plt.show()

grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), indexing='ij')
x_values = x_values / np.max(x_values)
y_values = y_values_unnorm / np.max(y_values_unnorm)

for i in range(len(x_values[0])):
    x = x_values[0][i]
    y = y_values[0][i]
    z = z_values[i]
    print(f"({x},{y},{z})")

points = np.hstack((x_values.T, y_values.T))
from scipy.interpolate import griddata
z_values = z_values*(z_values>minlim)*(z_values<maxlim)
grid_z0 = griddata(points, z_values, (grid_x, grid_y), method='linear') # method has to be one of: ["nearest","linear","cubic"]



#plt.subplot(121)
plt.imshow(grid_z0.T, extent=(0, 1, 0, 1), origin='lower')
ytickslen = 27
plt.xlabel(r"adiabatic time ratio: $\frac{T_\mathrm{RA}}{T}$")
plt.ylabel(r"total time: $T$")
plt.yticks(np.linspace(0,1,ytickslen), np.linspace(0,int(np.max(y_values_unnorm[0]))+1, ytickslen)) #, rotation='vertical')
plt.colorbar()
plt.show()
#plt.subplot(122)

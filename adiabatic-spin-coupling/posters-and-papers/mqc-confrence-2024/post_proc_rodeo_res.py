import numpy as np
import matplotlib.pyplot as plt

#filename = "rodeo_results_temp.txt"
filename = "rodeo_results_temp_2_sigma3.txt"
#filename = "rodeo_results_temp_1_sigma1.txt"
with open(filename, "r") as f:
    rall = f.readlines()

xs = [float(i.rstrip()) for i in rall[0].split(",")]
ys = []
for i in range(len(xs)):
    ys.append([])

i = 1
MAX = len(rall)
while i<MAX:
    nextline = [float(e.rstrip()) for e in rall[i].split(",")]

    for j in range(len(nextline)):
        ys[j].append(nextline[j])

    i+=1

ys_av = []
ys_std_l = []
ys_std_u = []
for i in range(len(ys)):
    ys_av.append(np.average(ys[i]))
    ys_std_l.append(np.percentile(ys[i],50)-np.percentile(ys[i],25))
    ys_std_u.append(np.percentile(ys[i],75)-np.percentile(ys[i],50))


plt.errorbar(xs,ys_av, np.array([ys_std_l, ys_std_u]), fmt='r^')
plt.show()

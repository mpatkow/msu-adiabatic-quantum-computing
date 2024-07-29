import numpy as np
import matplotlib.pyplot as plt
import os


data = {}
#categories = ["e22.","e22222222.","e4444."]
#categories = ["e22.","e44.","e88.","e2222222222222222.","e44444444."]
categories = ["e22.","e88.","e44.","e22222222."]
#categories = ["e22.","e44.","e88.","e2222222222222222.","e44444444."]
EPSILON_RODEO = 0.01
for category in categories:
    data[category] = {}
    category_dataset = data[category]
    category_dataset['total_runtimes'] = []
    category_dataset['overlap_at_end'] = []
    category_dataset['estimated_cost_adiabatic_rodeo'] = []
    category_dataset['estimated_cost_rodeo_only'] = []
    category_dataset['estimated_cost_adiabatic_rodeo_2'] = []
    category_dataset['estimated_cost_rodeo_only_2'] = []
    filenames_to_analyze = []
    for filename in os.listdir("results/"):
        if category not in filename:
            continue
        else:
            filenames_to_analyze.append(filename)
            category_dataset['total_runtimes'].append(float(filename.split("_")[1][1:].replace("p",".")))

    get_T_param = lambda filename : float(filename.split("_")[1][1:].replace("p","."))
    filenames_to_analyze = sorted(filenames_to_analyze, key=get_T_param)
    category_dataset['total_runtimes'].sort()

    for filename in filenames_to_analyze:
        with open("results/"+filename, "r") as file_opened:
            
            raw_data = file_opened.read()
            raw_data = raw_data.replace("array", "np.array")
            run_data = eval(raw_data)

            category_dataset['overlap_at_end'].append(run_data['overlap'][-1].real)

            # Calculate the estimated cost. That is 1/overlap * adiabatic_time
            category_dataset['estimated_cost_adiabatic_rodeo'].append(run_data['t'][-1] * 1/run_data['overlap'][-1])
            #print(f"Estimated cost for applying rodeo after a single [2]*n -> [2n] adiabatic fusion: {estimated_cost_adiabatic_rodeo}")

            # Calculate the estimated cost for only rodeo. That is 1/(overlap (t=0))
            category_dataset['estimated_cost_rodeo_only'].append(1/run_data['overlap'][0])
            #print(f"Estimated cost for applying rodeo to initial state of [2]*n: {estimated_cost_rodeo_only}")

            # Calculate the estimated cost by new method.  
            a_sq_end = run_data['overlap'][-1]
            N_rodeo_end = max(1,np.log2(1/EPSILON_RODEO * (1/a_sq_end - 1)))
            a_sq_start = run_data['overlap'][0]
            N_rodeo_start = max(1,np.log2(1/EPSILON_RODEO * (1/a_sq_start - 1)))
            category_dataset['estimated_cost_adiabatic_rodeo_2'].append(run_data['t'][-1] * N_rodeo_end/a_sq_end)
            category_dataset['estimated_cost_rodeo_only_2'].append(N_rodeo_start/a_sq_start)

plt.subplot(1,2,1)
for category in data.keys():
    run_dataset = data[category]
    plt.plot(run_dataset['total_runtimes'], sorted(run_dataset['overlap_at_end']), linestyle="dashed", label=category)
plt.legend()
plt.xlabel(r"Total runtime $T$")
plt.ylabel(r"Overlap $|\langle \psi _0 | \phi \rangle |^2$")

plt.subplot(1,2,2)
for category in data.keys():
    run_dataset = data[category]
    plt.plot(run_dataset['total_runtimes'], np.divide(run_dataset['estimated_cost_adiabatic_rodeo'], run_dataset['estimated_cost_rodeo_only']), label=category+"_original_method")
    plt.plot(run_dataset['total_runtimes'], np.divide(run_dataset['estimated_cost_adiabatic_rodeo_2'], run_dataset['estimated_cost_rodeo_only_2']), label=category+"_including rodeo cycles", linestyle = "dashed")
    #plt.plot(run_dataset['total_runtimes'], run_dataset['estimated_cost_rodeo_only_2'], label=category+"_rodeo_2_cost")
    #plt.plot(run_dataset['total_runtimes'], run_dataset['estimated_cost_adiabatic_rodeo_2'], label=category+"_adiabatic_rodeo_2")
plt.axhline(1, linestyle='dotted',color="black")
plt.legend()
plt.xlabel(r"Total runtime $T$")
plt.ylabel(r"Adiabatic Rodeo Cost / Rodeo Only Cost")
plt.show()

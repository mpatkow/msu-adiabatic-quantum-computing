import os
import tqdm

datadir = "data"

for filename in tqdm.tqdm(os.listdir(datadir)):
    with open(datadir + "/" + filename, "r") as f:
        r = f.readlines()
        if "cost" in r[-2]:
            continue
    os.system(f"python adiabatic_rodeo.py {datadir + '/' + filename}")

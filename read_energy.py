import matplotlib.pyplot as plt
import numpy as np
from opt_einsum import contract
import sys
import os


dir = sys.argv[1]

energy = np.zeros(int(sys.argv[2]))
Sz = np.zeros(int(sys.argv[2]))

h = np.linspace(float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[2]))


directory = os.fsencode(dir)


for sub_dir in os.listdir(directory):
    print(sub_dir)
    ind = int(sub_dir.decode("utf-8").split("_")[-1])-1
    print(ind)
    files = os.fsdecode("./"+dir+"/"+sub_dir.decode("utf-8")+"/output/")
    for filename in os.listdir(files):
        if filename.endswith("zvo_energy.dat"):
            test = open(files+filename, "r")
            energy[ind]=(float(test.read().split()[3]))
        if filename.endswith("Sz.dat"):
            test = open(files+filename, "r")
            Sz[ind]=float(test.read())

plt.plot(h, energy)
plt.savefig("./energy_Jpm=" + sys.argv[5] + ".png")
plt.clf()

plt.plot(h, np.gradient(np.gradient(energy, h),h))
plt.savefig("./energy_2nd_derivative" + sys.argv[5] + ".png")
plt.clf()


plt.plot(h, Sz)
plt.savefig("./Sz_Jpm=" + sys.argv[5] + ".png")
plt.clf()

plt.plot(h, np.gradient(np.gradient(Sz, h),h))
plt.savefig("./Sz_2nd_derivative" + sys.argv[5] + ".png")
plt.clf()
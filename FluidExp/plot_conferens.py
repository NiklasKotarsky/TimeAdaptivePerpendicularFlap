import numpy as np 
from pathlib import Path
import os
from matplotlib import pyplot as plt
import re

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 18
plt.rcParams['lines.markersize'] = 12

p = Path('.')

solutions = []

rel_errors_dis = []
rel_error_f = []
TimeWindows = []
thetas = []


target_file = "precice-Solid-convergenceQN.log"

pipefile=open(target_file,'r')
lines = pipefile.readlines()

e_QN_dis = []
T_W = []
i = 0
for line in lines:
    l = line.split()
    print(l)
    if i != 0:
        T_W.append(float(l[0]))            
        e_QN_dis.append(float(l[2]))
    i += 1
pipefile.close()


target_file = "precice-Solid-convergenceRel.log"

pipefile=open(target_file,'r')
lines = pipefile.readlines()

e_Rel_dis = []
T_W = []
i = 0
for line in lines:
    l = line.split()
    print(l)
    if i != 0:
        T_W.append(float(l[0]))            
        e_Rel_dis.append(float(l[2]))
    i += 1
pipefile.close()



plt.figure()
plt.title("Convergence speed")
plt.xlabel("Iterations")
plt.ylabel("Relative residual")
plt.semilogy(e_QN_dis,'*-',label = "QN Multirate")
plt.semilogy(e_Rel_dis,'o-',label = "Multirate")
plt.legend()
plt.savefig("ConvSpeedFirstTimeWindow", bbox_inches="tight")















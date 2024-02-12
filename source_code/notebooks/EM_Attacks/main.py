#!/usr/bin/python3
import qcircuits as qc
import numpy as np
import library as lib
import sys
import time

s = []

for i in range(0,15):
    phi = lib.SimonEM()
    print("********** phi ************") #10 qubit tensor
    print(phi)
    v = phi.to_column_vector()
    x = lib.QubitToVector(v)
    #print("*******x before ******\n", x)
    while(len(x) < 12):
        x = x[:2] + '0' + x[2:]
        #print("\n*******x after******\n", x)
    s.append([int(d) for d in x[2:]])
    #print("\n********s array*******\n")
    #print(s)
M = np.array(s,dtype=object)
#print("\n********Reduced Matrix*******\n")
#print(M)
B = lib.gauss(M)
#print("\n********Gaussian Elimination*******\n")
#print(B)
C = lib.gaussH(B)
#print("\n********Hermitian*******\n")
#print(C)
N  = lib.ReduceM(C)
#print("\n********Reduced Matrix*******\n")
#print(N)
if(len(N) < 9):
    print("No solution")
    sys.exit(False)
else:
    k = lib.solver(N)
    print(k)
    sys.exit(True)

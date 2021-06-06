#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import taishoten as tn

Si = np.arange(0,3) 
Sj = np.arange(0,4)
Sk = np.arange(1,5)
Sl = np.arange(0,2) 
Sm = np.arange(0,5)

dim = 20

symA = tn.Symmetry1D("++-", [Si,Sj,Sk])
symB = tn.Symmetry1D("++-", [Sk,Sl,Sm])

arrayA = np.random.randn(len(Si), len(Sj), dim, dim, dim)
arrayB = np.random.randn(len(Sk), len(Sl), dim, dim, dim)

A = tn.Tensor(arrayA, symA)
B = tn.Tensor(arrayB, symB)
C = tn.symeinsum("ijk,klm->ijlm", A, B)

print("Success! C.shape = {}, C.sym.fullsigns = {}".format(C.shape, C.sym.fullsigns)))































































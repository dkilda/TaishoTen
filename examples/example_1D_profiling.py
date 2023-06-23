#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
sys.path.insert(0, '..')

import numpy as np
import numpy.testing as nptest

import taishoten as tn
from taishoten import Str

import timeit


Si = np.arange(0,3) 
Sj = np.arange(0,4)
Sk = np.arange(1,7)
Sl = np.arange(0,2) 
Sm = np.arange(0,5)

dim = 20

symA = tn.Symmetry1D("++-", [Si,Sj,Sk])
symB = tn.Symmetry1D("++-", [Sk,Sl,Sm])

np.random.seed(1)
A = tn.random([len(Si), len(Sj), dim, dim, dim], symA) 
B = tn.random([len(Sk), len(Sl), dim, dim, dim], symB)


def symeinsum():
    C = tn.symeinsum("ijk,klm->ijlm", A, B)
    symC = C.sym
    return C

C     = symeinsum()
mapC  = tn.Map.compute(C.sym, Str("IJLM"))
Afull = A.get_full_array()
Bfull = B.get_full_array()

def np_einsum():
    arrayC = np.einsum("IJKijk,KLMklm->IJLMijlm", Afull, Bfull)
    return arrayC

#arrayC = np.einsum("IJLMijlm,IJLM->IJLijlm",  arrayC, mapC.array)


cpu_time_2 = timeit.timeit(symeinsum, number=1)
print("Symeinsum: ", cpu_time_2)

cpu_time_1 = timeit.timeit(np_einsum, number=1)
print("np einsum: ", cpu_time_1)

print("Ratio: ", cpu_time_1/cpu_time_2)




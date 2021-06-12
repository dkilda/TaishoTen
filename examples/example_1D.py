#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
sys.path.insert(0, '..')

import numpy as np
import numpy.testing as nptest

import taishoten as tn
from taishoten import Str


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

C = tn.symeinsum("ijk,klm->ijlm", A, B)

mapC   = tn.Map.compute(C.sym, Str("IJLM"))
arrayC = np.einsum("IJKijk,KLMklm->IJLMijlm", A.get_full_array(), B.get_full_array())
arrayC = np.einsum("IJLMijlm,IJLM->IJLijlm",  arrayC, mapC.array)


print("A.shape = {}, B.shape = {}, C.shape = {}".format(A.shape,       B.shape,       C.shape))
print("A.shape = {}, B.shape = {}, C.shape = {}".format(A.array.shape, B.array.shape, C.array.shape))

print("Success! C.shape = {}, C.sym.fullsigns = {}".format(C.shape, C.sym.fullsigns))

print("assert arrays       equal: ", nptest.assert_allclose(C.array, arrayC, rtol=1e-6, atol=1e-12))
print("assert array shapes equal: ", C.array.shape, arrayC.shape)

diff = np.linalg.norm(C.array - arrayC) / np.sqrt(arrayC.size)
print("assert array diff norm: ", diff < 1e-8, diff)


























































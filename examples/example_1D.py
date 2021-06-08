#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
sys.path.insert(0, '..')

import numpy as np
import taishoten as tn
from taishoten import Str

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


legs   = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
symcon = tn.compute_symmetry_contraction(symA, symB, legs)
symC   = symcon.sym("C")
mapC   = tn.Map.compute(symC, symcon.symlegs("C"))

A = tn.Tensor(arrayA, symA)
B = tn.Tensor(arrayB, symB)
C = tn.symeinsum("ijk,klm->ijlm", A, B)

arrayC = np.einsum("IJKijk,KLMklm->IJLMijlm", A.get_full_array(), B.get_full_array())
arrayC = np.einsum("IJLMijlm,IJLM->IJLijlm",  arrayC, mapC.array)

print("A.shape = {}, B.shape = {}, C.shape = {}".format(A.shape, B.shape, C.shape))
print("A.shape = {}, B.shape = {}, C.shape = {}".format(A.array.shape, B.array.shape, C.array.shape))

print("Success! C.shape = {}, C.sym.fullsigns = {}".format(C.shape, C.sym.fullsigns))


import numpy.testing as nptest
print("assert arrays       equal: ", nptest.assert_allclose(C.array, arrayC, rtol=1e-8, atol=1e-16))
print("assert array shapes equal: ", C.array.shape, arrayC.shape)



























































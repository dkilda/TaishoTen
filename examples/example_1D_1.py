#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
sys.path.insert(0, '..')

import numpy as np
import numpy.testing as nptest

import taishoten as tn
from taishoten import Str


Si = range(0,2)
no, nv = 5, 8
nmode = 3

symA = tn.Symmetry1D("++-0-", [Si]*4)
symB = tn.Symmetry1D("++-0-", [Si]*4)

np.random.seed(1)
A = tn.random([len(Si), len(Si), len(Si), no,no,nv,nmode,nv], symA) 
B = tn.random([len(Si), len(Si), len(Si), nv,nv,nv,nmode,nv], symB)

sub  = "ijcxd,abcxd->ijab"
sub1 = "IJCDijcxd,ABCDabcxd->IJABijab"
sub2 = "IJABijab,IJAB->IJAijab"


C = tn.symeinsum(sub, A, B)

mapC   = tn.Map.compute(C.sym, Str("IJAB"))
arrayC = np.einsum(sub1, A.get_full_array(), B.get_full_array())
arrayC = np.einsum(sub2, arrayC, mapC.array)



print("A.shape = {}, B.shape = {}, C.shape = {}".format(A.shape,       B.shape,       C.shape))
print("A.shape = {}, B.shape = {}, C.shape = {}".format(A.array.shape, B.array.shape, C.array.shape))

print("Success! C.shape = {}, C.sym.fullsigns = {}".format(C.shape, C.sym.fullsigns))

print("assert array shapes equal: ", C.array.shape, arrayC.shape)
print("assert arrays       equal: ", nptest.assert_allclose(C.array, arrayC, rtol=1e-6, atol=1e-12))

diff = np.linalg.norm(C.array - arrayC) / np.sqrt(arrayC.size)
print("assert array diff norm: ", diff < 1e-8, diff)












































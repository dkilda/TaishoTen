#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
sys.path.insert(0, '..')

import itertools
import numpy as np
import numpy.testing as nptest

import taishoten as tn
from taishoten import Str



def make_symlabels_3D(Gvecs, num_pts):

    # Generate lists of symlabels along x,y,z axes \in (0,1)
    # --> find  all their combos (triples) using Cartesian product
    # --> gives scaled symlabels = list of all possible triples
    symlabel_arrays  = [np.arange(n) / n for n in num_pts]
    scaled_symlabels = [np.asarray(s) \
                        for s in itertools.product(*symlabel_arrays)]

    scaled_symlabels = np.asarray(scaled_symlabels)

    # Multiply scaled symlabels by Gvecs
    # scaled_symlabels \in (0,1) -> symlabels \in (0,G)
    symlabels = np.dot(scaled_symlabels, Gvecs)
    return symlabels
    


def get_reciprocal_vecs(lattice_vecs):

    # From matrix of lattice vectors, get matrix of reciprocal vectors 
    return 2 * np.pi * np.linalg.inv(lattice_vecs.T)
 


# Make kpts
lattice_vecs = 5 * np.eye(3)
gvecs = get_reciprocal_vecs(lattice_vecs)
kpts  = make_symlabels_3D(gvecs, [2,2,1])

nkpts = len(kpts)
dim   = 10
shape = (nkpts, dim, dim)


# Setup symmetries and tensors
sym = tn.Symmetry3D("+-", [kpts]*2, mod=gvecs)

np.random.seed(1)
A = tn.random(shape, sym)
B = tn.random(shape, sym)


# Do symmetric einsum
C = tn.symeinsum("ij,jk->ik", A, B)

mapC   = tn.Map.compute(C.sym, Str("IK"))
arrayC = np.einsum("IJij,JKjk->IKik", A.get_full_array(), B.get_full_array())
arrayC = np.einsum("IKik,IK->Iik", arrayC, mapC.array)


print("A.shape = {}, B.shape = {}, C.shape = {}".format(A.shape, B.shape, C.shape))
print("A.shape = {}, B.shape = {}, C.shape = {}".format(A.array.shape, B.array.shape, C.array.shape))

print("Success! C.shape = {}, C.sym.fullsigns = {}".format(C.shape, C.sym.fullsigns))

print("assert arrays       equal: ", nptest.assert_allclose(C.array, arrayC, rtol=1e-6, atol=1e-12))
print("assert array shapes equal: ", C.array.shape, arrayC.shape)

diff = np.linalg.norm(C.array - arrayC) / np.sqrt(arrayC.size)
print("assert array diff norm: ", diff < 1e-8, diff)



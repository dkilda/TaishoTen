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

import timeit


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
shape = (nkpts, nkpts, nkpts, dim, dim, dim, dim)


# Setup symmetries and tensors
sym = tn.Symmetry3D("++--", [kpts]*4, mod=gvecs)

np.random.seed(1)
A = tn.random(shape, sym)
B = tn.random(shape, sym)


def symeinsum():
    C = tn.symeinsum("ijab,abcd->ijcd", A, B)
    return C

C     = symeinsum()
mapC  = tn.Map.compute(C.sym, Str("IJCD"))
Afull = A.get_full_array()
Bfull = B.get_full_array()

def np_einsum():
    arrayC = np.einsum("IJABijab,ABCDabcd->IJCDijcd", Afull, Bfull)
    return arrayC

#arrayC = np.einsum("IJCDijcd,IJCD->IJCijcd", arrayC, mapC.array)


cpu_time_2 = timeit.timeit(symeinsum, number=1)
print("Symeinsum: ", cpu_time_2)

cpu_time_1 = timeit.timeit(np_einsum, number=1)
print("np einsum: ", cpu_time_1)

print("Ratio: ", cpu_time_1/cpu_time_2)




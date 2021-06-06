#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import taishoten as tn


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


# Setup symmetries and arrays
sym = tn.Symmetry3D("++--", [kpts]*4, mod=gvecs)

arrayA = np.random.randn(*shape)
arrayB = np.random.randn(*shape)


# Do symmetric einsum
A = tn.Tensor(arrayA, sym)
B = tn.Tensor(arrayB, sym)
C = tn.symeinsum("ijab,abcd->ijcd", A, B)


print("Success! C.shape = {}, C.sym.fullsigns = {}".format(C.shape, C.sym.fullsigns)))




















































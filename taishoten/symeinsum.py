#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools
 
import taishoten.transformations as trans
import taishoten.util as util

from taishoten.util import StrSet
from taishoten.util import assertequal
from taishoten.util import IS, ISNOT, ARE, ARENOT

from taishoten.symmetry import SymmetryContractor
from taishoten.tensor import Tensor






def symeinsum(subscript, tensA, tensB):

    # Prelims: get backend
    msg = "symeinsum: tensA and tensB must have the same backend"
    assertequal(tensA.backend, tensB.backend, msg)
    backend = tensA.backend

    # Prelims: preprocess subscript, get dense legs from subscript
    subscript = preprocess_subscript(subscript)
    denselegs = util.subscript_to_legs(subscript)

    # (1) Trivial case: no symmetry
    if  ISNOT(tensA.sym) or ISNOT(tensB.sym):
        arrayC = backend.einsum(tensA.array, tensB.array)
        return Tensor(arrayC, backend=backend)


    # (2) Contract symmetries of tensA and tensB, compute all maps
    symcon  = SymmetryContractor(tensA.sym. tensB.sym, denselegs) 
    symC    = symcon.compute_output_sym()    
    symlegs = symcon.symlegs()                                 
    maps    = compute_maps(symcon, backend)


    # (3) Direct contraction
    if  trans.good_to_contract(symlegs): 

        full_subscript = util.sym_dense_legs_to_subscript(symlegs, denselegs)
        arrayC = backend.einsum(full_subscript, tensA.array, tensB.array)

        return Tensor(arrayC, symC, backend=backend)


    # (4) Direct contraction not possible, transformations are needed
    path = trans.find_optimal_transform_path(maps, symlegs)

    tensA.transform(path[0], denselegs[0])
    tensB.transform(path[1], denselegs[1])

    arrayC = backend.einsum(full_subscript, tensA.array, tensB.array)

    tensC = Tensor(arrayC, symC, backend=backend)
    tensC = tensC.transform(path[2], denselegs[2])

    return tensC





def preprocess_subscript(subscript):

    # Make sure subscript is in lowercase
    subscript = subscript.lower()

    # Add arrow if not present
    if  '->' not in subscript:
        subscript += '->'

    # Symbol q is reserved for aux legs
    if  'q' in subscript:
        msg = "symeinsum: q must only be used for auxiliary legs"
        raise ValueError(msg)

    return subscript











































































































































































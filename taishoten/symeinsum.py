#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools
 
import taishoten.transformations as trans
import taishoten.util as util

from taishoten.util import Str, dictriplet
from taishoten.util import assertequal
from taishoten.util import IS, ISNOT, ARE, ARENOT

from taishoten.symmetry import compute_symmetry_contraction, compute_maps
from taishoten.tensor   import Tensor




def symeinsum(subscript, tensA, tensB):

    # Prelims: get backend
    msg = "symeinsum: tensA and tensB must have the same backend"
    assertequal(tensA.backend, tensB.backend, msg)
    backend = tensA.backend

    # Prelims: preprocess subscript, get dense legs from subscript
    subscript = preprocess_subscript(subscript)
    legs      = util.subscript_to_legs(subscript)
    legs      = dictriplet(*legs)


    # (1) Trivial case: no symmetry
    if  ISNOT(tensA.sym) or ISNOT(tensB.sym):
        arrayC = backend.einsum(tensA.array, tensB.array)
        return Tensor(arrayC, backend=backend)


    # (2) Contract symmetries of tensA and tensB, compute all maps
    symcon  = compute_symmetry_contraction(tensA.sym, tensB.sym, legs) 
    symC    = symcon.sym("C")   
    symlegs = symcon.symlegs()                                 
    maps    = compute_maps(symcon, backend)


    # (3) Direct contraction
    if  trans.good_to_contract(symlegs): 

        full_subscript = make_full_subscript(symlegs, legs)
        arrayC = backend.einsum(full_subscript, tensA.array, tensB.array)
        return Tensor(arrayC, symC, backend=backend)


    # (4) Direct contraction not possible, transformations are needed
    path, final_symlegs = trans.find_transform_path(maps, symlegs)

    tensA = tensA.transform(path["A"], legs["A"])
    tensB = tensB.transform(path["B"], legs["B"])

    full_subscript = make_full_subscript(final_symlegs, legs, truncated=True)
    arrayC         = backend.einsum(full_subscript, tensA.array, tensB.array)

    tensC = Tensor(arrayC, symC, backend=backend)
    tensC = tensC.transform(path["C"], legs["C"])
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




def make_full_subscript(symlegs, legs, truncated=False):

    # Construct a full list of legs, both symmetric and dense ones
    full_legs = []
    for key in legs.keys():

         ll = legs[key]
         ss = symlegs[key] 
 
         if not truncated:
            ss = util.truncate(ss)

         full_legs.append(ss + ll)

    # Convert them to einsum subscript
    subscript = util.legs_to_subscript(*full_legs) 
    return subscript





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools

from . import backends

from .util import Str
from .util import assert_equal, isiterable, del_from_list, multisorted
from .util import subscript_to_legs, legs_to_subscript
from .util import symlegs_and_denselegs_to_subscript

from .symmetry        import SymConBuilder
from .transformations import find_optimal_transform_sequence
from .tensor          import Tensor 




def symeinsum(subscript, tensA, tensB):

    # Prelims: get the backend
    backend = tensA.backend
    assert_equal(tensA.backend, tensB.backend, \
                 "sym_einsum: tensA and tensB must have the same backend")

    # Prelims: preprocess subscript
    subscript = preprocess_subscript(subscript)

    
    # (1) --- Simple case: no symmetry ---

    if  tensA.sym is None:

        # Simple multiplication of A and B tensors, no symmetry
        array_C = backend.einsum(tensA.get_array(), tensB.get_array())
        return Tensor(array_C)


    # (2) --- Symmetric case: preparation ---

    # Build symcon depot, which is responsible for managing a symmetric contraction
    symcon = SymConBuilder(subscript, tensA, tensB)

    # Get denselegs, symlegs and output symmetry
    denselegs = symcon.get_denselegs()
    symlegs   = symcon.get_symlegs()
    symC      = symcon.get_output_symmetry()


    # (3) --- Symmetric case: direct contraction ---

    if  good_to_contract(symlegs):

        # Make subscript, contract A and B
        full_subscript = symlegs_and_denselegs_to_subscript(symlegs, denselegs)
        arrayC         = backend.einsum(full_subscript, tensA.get_array(), tensB.get_array())

        return Tensor(arrayC, symC)


    # (4) --- Symmetric case: direct contraction not possible, requires transformations ---

    # Find the optimal transformation sequence using symcon depot from above
    transform_sequence_ABC, full_subscript = find_optimal_transform_sequence(symcon)

    # Transform A and B to a directly contractible configuration
    tensA = tensA.transform(transform_sequence_ABC["A"], denselegs["A"])
    tensB = tensB.transform(transform_sequence_ABC["B"], denselegs["B"])

    # Conctract A and B (since now they're in a contractible config)
    arrayC = backend.einsum(full_subscript, tensA.get_array(), tensB.get_array())

    # Transform C back to the original configuration
    tensC = Tensor(arrayC, symC)
    tensC = tensC.transform(transform_sequence_ABC["C"], denselegs["C"])
    return tensC





def preprocess_subscript(subscript):

    # Make sure subscript is in lowercase
    subscript = subscript.lower()

    # Add arrow if not present
    if  '->' not in subscript:
        subscript += '->'

    # Symbol q is reserved for aux legs
    if  'q' in subscript:
        raise ValueError("sym_einsum: q must only be used for auxiliary legs")

    return subscript



































































































































































































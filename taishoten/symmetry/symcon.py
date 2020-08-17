#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

import numpy as np
import copy  as cp
import itertools

from ..util import Str
from ..util import assert_equal, isiterable, del_from_list, multisorted
from ..util import NotImplementedField
from ..util import subscript_to_legs, legs_to_subscript
from ..util import symlegs_and_denselegs_to_subscript

from .symmetry          import SymFactory
from .symcon_irrep_maps import SymConWithIrrepMaps



def SymConBuilder(subscript, tensA, tensB):

    # Get tensor symmetries, their dense legs and dense shapes
    symA = tensA.get_sym()
    symB = tensB.get_sym()

    dense_shape_A = tensA.get_dense_shape()
    dense_shape_B = tensB.get_dense_shape()

    denselegs = subscript_to_legs(subscript)

    # Make a symmetric contraction object
    symcon = SymConFactory(symA, symB, denselegs)

    # Build symcon: compute all the maps and their legs, 
    # calculate output symmetry, get dims of symlegs and denselegs
    symcon.compute_maps()
    symcon.compute_output_sym()
    symcon.make_denseleg_dims(subscript, dense_shape_A, dense_shape_B)
    symcon.make_symleg_dims()

    # Return the freshly built symcon
    return symcon





def SymConFactory(symA, symB, denselegs):

    # Make sure symA and symB dimensions match
    assert_equal(symA.ndim, symB.ndim, "SymConFactory: symA and symB must have the same dimensionality") 
    ndim = symA.ndim

    if   ndim == 1:
         # 1D symmetries
         return SymCon1D(symA, symB, denselegs)

    elif ndim == 3:
         # 3D symmetries
         return SymCon3D(symA, symB, denselegs)

    else:
         # Invalid symmetries
         raise ValueError("SymConFactory: invalid symrange = {}".format(symrange))





class SymCon1D(SymCon):

  def __init__(self, symA, symB, denselegs):
 
      # Create 1D SymCon
      assert_equal(symA.ndim, 1, "SymCon1D: symA must be 1D")
      assert_equal(symB.ndim, 1, "SymCon1D: symB must be 1D")

      super().__init__(symA, symB, denselegs)





class SymCon3D(SymCon):

  def __init__(self, symA, symB, denselegs):

      # Create 3D SymCon
      assert_equal(symA.ndim, 3, "SymCon3D: symA must be 3D")
      assert_equal(symB.ndim, 3, "SymCon3D: symB must be 3D")

      super().__init__(symA, symB, denselegs)




































































#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools

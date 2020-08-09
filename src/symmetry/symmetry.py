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

from ..params import SYMTOL

from .symmetry_irrep_maps import SymmetryWithIrrepMap



def SymFactory(signs, symrange, qtot=0, mod=None, backend=None, tol=SYMTOL):

    if   not isiterable(symrange[0][0]):

         # Noniterable symrange elements: 1D symmetry
         return Sym1D(signs, symrange, qtot, mod, backend, tol)

    elif len(symrange[0][0]) == 3:

         # Iterable symrange elements with size=3: 3D symmetry
         return Sym3D(signs, symrange, qtot, mod, backend, tol)

    else:
         # Invalid symmetry specified
         raise ValueError("SymFactory: invalid symrange = {}".format(symrange))






class Sym1D(SymmetryWithIrrepMap):

  def __init__(self, signs, symrange, qtot=0, mod=None, backend=None, tol=SYMTOL):

      # Parent symmetry
      super().__init__(signs, symrange, qtot, mod, backend, tol)

      # Set symmetry dim
      self._ndim = 1


  # --- Auxiliary methods for symrange computation ----------------------------------------------------------

  def apply_mod(self, symrange):

      # If mod is None: return immediately
      if  self.mod is None:
          return symrange

      # Apply mod to symlabels in a given symrange
      symrange = np.mod(symrange, self.mod)
      return symrange


  def fold(self, symrange):

      # Fold symrange: apply mod and return [sorted unique symlabels]
      symrange = self.apply_mod(symrange)
      symrange = np.unique(symrange)
      return symrange


  @staticmethod
  def _make_symrange(symrange):

      # Make sure all symrange elements are non-iterable
      assert all(not isiterable(elem) for srng in symrange for elem in srng)

      # Process symrange input
      symrange = [np.asarray(srng) for srng in symrange]
      symrange =  np.asarray(symrange)
      return symrange

  # ---------------------------------------------------------------------------------------------------------








class Sym3D(SymmetryWithIrrepMap):

  def __init__(self, signs, symrange, qtot=0, mod=None, backend=None, tol=SYMTOL):

      # Parent symmetry
      super().__init__(signs, symrange, qtot, mod, backend, tol)

      # Set the symmetry dim
      self._ndim = len(self._symrange[0][0])
      assert_equal(self._ndim, 3, "Sym3D: the symmetry must be 3D")

      # Check the modulus size is consistent with the symmetry dim
      if  self._mod is not None:
          assert_equal(self._ndim, len(self._mod), \
                       "Sym3D: the symmetry dimension must be equal to the dimension of modulus"


  # --- Auxiliary methods for symrange computation ----------------------------------------------------------


  def apply_mod(self, symrange, folding=False):

      # If mod is None: return immediately
      if  self.mod is None:
          return symrange

      # Apply inverse mod matrix to symlabels in a given symrange
      inv_mod  = inverted_modulus_matrix_3x3(self.mod)
      symrange = np.dot(symrange, inv_mod)
      symrange = symrange - round_to_int(symrange, folding)

      return symrange


  def fold(self, symrange):

      # Fold symrange: apply mod and return [sorted unique symlabels]
      #    (for ndim > 1, symrange elements are vectors: 
      #     so we get unique symrange vectors along axis = 0)
      symrange = self.apply_mod(symrange)
      symrange = round_to_float(symrange) 
      symrange = np.unique(symrange, axis=0)

      return symrange


  @staticmethod
  def _make_symrange(symrange):

      # Make sure all symrange elements are arrays
      symrange = [np.asarray(elem) for srng in symrange for elem in srng]

      # Check all symrange elements have equal size
      ndim = len(symrange[0][0])
      assert all(len(elem) == ndim  for srng in symrange for elem in srng)

      # Process symrange input
      symrange = [np.asarray(srng) for srng in symrange]
      symrange =  np.asarray(symrange)
      return symrange

  # ---------------------------------------------------------------------------------------------------------









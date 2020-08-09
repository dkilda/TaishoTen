#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

import numpy as np
import copy  as cp
import itertools

import backends

from util import Str
from util import assert_equal, isiterable, del_from_list, multisorted
from util import NotImplementedField
from util import subscript_to_legs, legs_to_subscript
from util import symlegs_and_denselegs_to_subscript

from params import SYMTOL

from .symmetry_base import Symmetry




class SymmetryWithIrrepMap(Symmetry):

  def __init__(self, signs, symrange, qtot=0, mod=None, backend=None, tol=SYMTOL):

      super().__init__(signs, symrange, qtot, mod)

      self._map     = None
      self._tol     = tol
      self._backend = backends.get_backend(backend)


  def __new__(cls, signs, symrange, qtot=0, mod=None, backend=None, tol=SYMTOL, *args, **kwargs):

      # Prevent Symmetry from instantiation
      if  cls is SymmetryWithIrrepMap:
          raise TypeError("SymmetryWithIrrepMap class cannot be instantiated")
      return object.__new__(cls, *args, **kwargs)



  # --- Get methods and properties --------------------------------------------------------------------------

  def get_map(self, sym=None):
      if  self._map is None:
          self.compute_map()
      return cp.deepcopy(self._map)

  def get_map_shape(self):
      return self._map.shape

  @property
  def backend(self):
      return self._backend


  # --- Compute map -----------------------------------------------------------------------------------------

  def compute_map(self):

      # Compute combo symrange by adding together all legs
      # --> we get combo_symrange[idx] = sgn1*q1[i] + sgn2*q2[j] + sgn3*q3[k] + ..., idx = combo(i,j,k,...) 
      flat_symrange  = self.make_flat_symrange()

      # Subtract Qtot from symrange, apply modulus, and take ABS value: 
      # --> we get combo_symrange[idx] = |(sgn1*q1[i] + sgn2*q2[j] + sgn3*q3[k] + ... - Qtot) % mod|, idx = combo(i,j,k,...)
      flat_symrange -= self.qtot 
      flat_symrange  = self.mod(flat_symrange)
      flat_symrange  = self.sum_over_elements(symrange)

      # Find all indices where combo_symrange is zero, 
      # i.e. the conservation law |(sgn1*q1[i] + sgn2*q2[j] + sgn3*q3[k] + ... - Qtot) % mod| = 0 is satisfied
      idx   = self.find_zeros(flat_symrange)
      shape = self.get_shape()  

      # Initialize map operator as [array of zeros],
      #                with [num of legs] = [num of symmetric legs] in sym.symrange 
      self._map = self.generate_map_from_indices(shape, idx)
      return self


  # --- Auxiliary functions ---------------------------------------------------------------------------------

  def generate_map_from_indices(self, shape, idx):

      # Generate the map itself: set op[i,j,k,...] = 1 at each idx = combo(i,j,k,...) 
      #                          where combo_symrange[idx] = 0 (i.e. all symlabels add up to zero)
      new_map = self.backend.zeros(shape)
      vals    = self.backend.ones(len(idx))
      new_map = self.backend(new_map, idx, vals)
      return new_map

  # ---------------------------------------------------------------------------------------------------------











































































































































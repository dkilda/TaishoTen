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

from .params import ALPHABET



class Tensor:

  def __init__(self, array, symmetry=None, backend=None):

      self._array   = array 
      self._sym     = symmetry
      self._backend = backends.get_backend(backend)
 
      if   self._sym is None: 
   
           # No symmetry: treat as a dense tensor
           self._ndim        = self._array.ndim
           self._dense_shape = self._array.shape

      else:
           # Num of signed and unsigned symlegs
           num_signed_legs   = self._sym.get_num_signed_legs()
           num_unsigned_legs = self._sym.get_num_unsigned_legs() 

           # Get ndim = num of symlegs (signed and unsigned) = num of denselegs, 
           #    shape = shape of a dense block only (i.e. denselegs shape) 
           self._ndim        = num_signed_legs + num_unsigned_legs 
           self._dense_shape = self._array.shape[max(num_signed_legs - 1, 0) : ]

           # Make sure backends match
           assert_equal(self._backend, self._sym.backend, \
                        "Tensor: tensor backend and symmetry backend must be the same")


  # --- Properties and get/set methods ----------------------------------------------------------------------

  @property
  def dense_shape(self):
      return cp.deepcopy(self._dense_shape)

  @property
  def ndim(self):
      return cp.deepcopy(self._ndim)

  @property
  def backend(self):
      return self._backend

  def get_sym(self):
      return cp.deepcopy(self._sym)

  def set_sym(self, sym):
      self._sym = sym
      return self

  def get_array(self):
      return cp.deepcopy(self._array)

  def get_map(self):
      if  self._sym is None:
          return None
      return self._sym.get_map()

  def get_shape(self):

      # If nosym, just get dense shape
      if  self._sym is None:
          return self.dense_shape
  
      # From dense shape, get the full shape
      full_shape = self._sym.get_full_shape(self.dense_shape)
      return full_shape



  # --- Symmetric operations: compute full/reduced form, transform using a set of maps, etc -----------------

  def compute_full_array(self):

      # If no symmetry: return array itself
      if  self._sym is None:
          return self.get_array() 

      # Get map, get reduced-to-full subscript
      subscript  = self._reduced_to_full_subscript()
      irrep_map  = self._sym.get_map()

      # Construct full array (no hidden legs)
      full_array = self.backend.einsum(subscript, self._array, irrep_map)
      return full_array  


  def symmetrize(self):

      # If no symmetry: return immediately
      if  self._sym is None:
          return self

      # Get map, compute full array (no hidden legs), get full-to-reduced subscript
      subscript  = self._full_to_reduced_subscript() 
      irrep_map  = self._sym.get_map()
      full_array = self.compute_full_array()
     
      # Compute reduced array
      self._array = self.backend.einsum(subscript, full_array, irrep_map)
      return self


  def transform(self, transform_sequence, denselegs):

      # Num of transformations
      num_transforms = len(transform_sequence) 

      # If num of required transformations is zero: trivial return
      if  num_transforms == 0:
          return self._as_new_tensor(self._array)

      # Construct symlegs list and maps list from the transform sequence
      symlegs_list = []
      maps_list    = []

      for i, transf in enumerate(transform_sequence):
          
          # First transformation: add starting legs config of the tensor 
          if  i == 0:
              symlegs_list += [transf.start_legs]

          # Add map legs and map operator
          symlegs_list += [transf.map_legs]
          maps_list    += [transf.get_map()]

          # Last transformation: add final legs config of the tensor
          if  i == num_transforms - 1:
              symlegs_list += [transf.end_legs]

      # Get dense legs list (no denselegs for maps, only for the initial and final tensor)
      denselegs_list     = [''] * len(symlegs_list)  
      denselegs_list[ 0] = denselegs 
      denselegs_list[-1] = denselegs

      # Construct subscript from symlegs and denselegs
      subscript = symlegs_and_denselegs_to_subscript(symlegs_list, denselegs_list)
      
      # Multiply tensor with all the maps on a transformation sequence
      new_array = self.backend.einsum(subscript, self._array, *maps_list)  
      return self._as_new_tensor(new_array)
  


  # --- Auxiliary methods -----------------------------------------------------------------------------------

  def _as_new_tensor(self, array, sym=None):

      if  sym is None:
          sym = self._sym

      new_tensor = Tensor(array, sym)
      return new_tensor


  def _reduced_to_full_subscript(self):

      # Reduced-to-full transformation subscript
      return self._make_self_transform_subscript(to_reduced=False)


  def _full_to_reduced_subscript(self):

      # Full-to-reduced transformation subscript
      return self._make_self_transform_subscript(to_reduced=True)


  def _make_self_transform_subscript(self, to_reduced):

      # Signs, num of dims (shortcut)
      signs = self._sym.signs
      ndim  = self._ndim

      # Make legs and pure legs, e.g. legs = "ab"
      legs     = Str(ALPHABET[:ndim])
      purelegs = legs.purify(signs)

      # Generate subscript, e.g. ABab,AB->Aab
      legs_FULL    = purelegs + legs
      legs_REDUCED = purelegs.truncate() + legs
      legs_MAP     = purelegs

      # Arrange legs in order
      if   to_reduced:
           legs_list = [legs_FULL, legs_MAP, legs_REDUCED] 
      else:
           legs_list = [legs_REDUCED, legs_MAP, legs_FULL] 

      # Convert to subcript and return
      subscript = legs_to_subscript(legs_list)
      return subscript

  # ---------------------------------------------------------------------------------------------------------






























































































































































































































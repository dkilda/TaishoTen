#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools

import taishoten.backends as backends
import taishoten.util as util

from taishoten.util     import StrSet
from taishoten.params   import ALPHABET
from taishoten.symmetry import Map


class Tensor:

  def __init__(self, array, sym=None, backend=None):

      self._backend = backends.get_backend(backend)
      self._sym     = sym
      self._array   = self.backend.asarray(array) 

      if   sym is None: 
   
           # No symmetry: treat as a dense tensor
           self._ndim        = array.ndim
           self._dense_shape = array.shape

      else:
           # Dense indices in the symmetric array 
           # (whether in its full or truncated form)
           dense_idx = slice(array.ndim - self._ndim, array.ndim)

           # Get ndim and dense shape:
           # Num of denselegs = num of signed legs + num of unsigned legs,
           # Num of symlegs   = num of signed legs
           self._ndim        = sym.num_signed_legs + sym.num_unsigned_legs
           self._dense_shape = array.shape[dense_idx]

           # Construct truncated form of Tensor._array
           self._make_truncated_array()


  # --- Auxiliary methods --------------------------------------------------- #

  def _as_new_tensor(self, array=None, sym=None):

      # Create a new Tensor object
      if  array is None:
          array = self._array

      if  sym is None:
          sym = self._sym

      return type(self)(array, sym)



  def _make_subscript(self, make_full=False):

      # Make dense and symmetric legs (ndim = num of dense legs)
      denselegs = Str(ALPHABET[:self.ndim])
      symlegs   = util.make_symlegs(denselegs, self.sym.signs)

      # Construct legs of the full form (e.g. ABab), 
      # the truncated form (e.g. Aab), and the map (e.g. AB)
      full_legs  = symlegs + denselegs
      trunc_legs = util.truncate(symlegs) + denselegs
      map_legs   = symlegs

      # Generate subscript,
      # e.g. ABab,AB->Aab (full-to-truncated transformation), 
      # or   Aab,AB->ABab (truncated-to-full transformation)
      if  make_full:
          subscript = util.legs_to_subscript(trunc_legs, map_legs, full_legs)
      else:
          subscript = util.legs_to_subscript(full_legs,  map_legs, trunc_legs)

      return subscript, map_legs
  


  def _make_truncated_array(self): 

      # Construct truncated form of Tensor._array
      trunc_ndim = 2*self.ndim - 1

      if  array.ndim > trunc_ndim:

          # Get full-to-truncated subscript
          subscript, map_legs = self._make_subscript()

          # Compute map from sym, compute symmetrized array 
          new_map     = Map.compute(self.sym, map_legs, self.backend)
          self._array = self.backend.einsum(subscript, \
                                            self._array, new_map.array)
      return self



  def _get_full_array(self): 

      # If no symmetry
      if  ISNOT(self.sym):
          return self._array

      # Get truncated-to-full subscript
      subscript, map_legs = self._make_subscript(make_full=True)

      # Compute map from sym, compute full array (no hidden legs)
      new_map    = Map.compute(self.sym, map_legs, self.backend)
      full_array = self.backend.einsum(subscript, \
                                       self._array, new_map.array)
      return full_array



  # --- Properties and get/set methods -------------------------------------- #

  @property
  def array(self):
      return self._array

  @property
  def backend(self):
      return self._backend

  @property
  def ndim(self):
      return self._ndim

  @property
  def sym(self):
      return self._sym

  @property
  def signs(self):
      if  ISNOT(self.sym):
          return None
      return self.sym.signs


  @property
  def shape(self):

      # If no sym, just get dense shape
      if  self._sym is None:
          return self.dense_shape

      # Combine symmetric and dense shapes
      full_shape = self._sym.shape + self.dense_shape
      return full_shape


  @property
  def dense_shape(self):
      return self._dense_shape


  def denseleg_dims(self, denselegs):
      return {leg: dim for leg, dim in zip(denselegs, self.dense_shape)}



  # --- Symmetric transformations ------------------------------------------- # 

  def transform(self, transform_path, denselegs):

      # Num of transformation nodes
      num_nodes = len(transform_path) 

      # (1) Trivial case: path with no transformations
      if  num_nodes == 0:
          return self._as_new_tensor(self._array)

      # (2) Construct a list of symlegs and maps 
      #     for transformation einsum
      symlegs_lst = [node.start_legs]
      maps_lst    = []

      for i, node in enumerate(transform_path): 

          # First node: add start legs of tensor
          if  i == 0:
              symlegs_lst += [node.start_legs]

          # Middle nodes: add map legs and map itself
          symlegs_lst += [node.map_legs]
          maps_lst    += [node.map_array] 

          # Last node: add end legs of tensor
          if  i == num_nodes - 1:
              symlegs_lst += [node.end_legs]

      # (3) Construct a list of denselegs. NB maps have no dense legs, 
      #     only the initial and final tensors do.
      denselegs_lst     = [StrSet()] * len(symlegs_lst)  
      denselegs_lst[ 0] = denselegs 
      denselegs_lst[-1] = denselegs 

      # (4) Construct einsum subscript from symlegs and denselegs     
      subscript = util.sym_dense_legs_to_subscript(symlegs_lst, denselegs_lst)

      # (5) Contract tensor with all the transformation maps
      new_array = self.backend.einsum(subscript, self._array, *maps_lst)  
      return self._as_new_tensor(new_array)
  

  # ------------------------------------------------------------------------- #






def get_denseleg_dims(tensA, tensB, denselegs_A, denselegs_B):

    return {**tensA.denseleg_dims(denselegs_A), \
            **tensB.denseleg_dims(denselegs_B)  } 

    





















































































































































































































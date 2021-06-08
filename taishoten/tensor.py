#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools

import taishoten.backends as backends
import taishoten.util as util

from taishoten.util import Str
from taishoten.util import assertequal
from taishoten.util import IS, ISNOT, ARE, ARENOT

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
           # Get ndim and dense shape of the array
           self._ndim        = sym.num_legs
           self._dense_shape = array.shape[-sym.num_legs : ] 

           # Symmetrized form of input array
           self._array = self.get_symmetrized_array(array)


  # --- Auxiliary methods --------------------------------------------------- #

  def as_new_tensor(self, array=None, sym=None):

      # Create a new Tensor object
      if  array is None:
          array = self._array

      if  sym is None:
          sym = self._sym

      return type(self)(array, sym)



  def _make_transform_legs(self): 

      # Make dense and symmetric legs (ndim = num of dense legs)
      denselegs = util.make_legs(self.ndim)
      symlegs   = util.make_symlegs(denselegs, self.sym.fullsigns)      

      # Construct legs of the full form (e.g. ABab), 
      # the truncated form (e.g. Aab), and the map (e.g. AB)
      full_legs  = symlegs + denselegs
      trunc_legs = util.truncate(symlegs) + denselegs
      map_legs   = symlegs

      return full_legs, trunc_legs, map_legs

      

  def get_symmetrized_array(self, array): 

      # Get symmetrized/truncated form of input array 
      # (only truncate if array.ndim > truncated array ndim)

      # Shortcut
      sym = self.sym

      # The ndim of truncated input array
      trunc_array_ndim = 2*self.ndim - 1

      # Check symmetry is consistent with array dims
      if   array.ndim == trunc_array_ndim:

           msg = "Tensor.get_symmetrized_array: "\
                 "array and symmetry truncated shapes must match"

           arr_shape = array.shape[: sym.num_symlegs - 1]
           assertequal(arr_shape, sym.truncated_shape, msg)

      elif array.ndim == trunc_array_ndim + 1:

           msg = "Tensor.get_symmetrized_array: "\
                 "array and symmetry shapes must match"

           arr_shape = array.shape[: sym.num_symlegs]
           assertequal(arr_shape, sym.shape, msg)

      else:
           msg = "Tensor.get_symmetrized_array: "\
                 "invalid ndim = {} of input array".format(array.ndim)
           raise ValueError(msg)


      # Input already truncated: trivial return
      if  array.ndim == trunc_array_ndim:
          return array

      # Make full-to-truncated subscript (e.g. ABab,AB->Aab)
      full_legs, trunc_legs, map_legs = self._make_transform_legs()           
      subscript = util.legs_to_subscript(full_legs, map_legs, trunc_legs) 

      # Compute map from sym, compute symmetrized array 
      new_map = Map.compute(sym, map_legs, self.backend)
      array   = self.backend.einsum(subscript, array, new_map.array)

      return array



  def get_full_array(self): 

      # If no symmetry
      if  ISNOT(self.sym):
          return self._array

      # Make truncated-to-full subscript (e.g. Aab,AB->ABab)
      full_legs, trunc_legs, map_legs = self._make_transform_legs()           
      subscript = util.legs_to_subscript(trunc_legs, map_legs, full_legs) 

      # Compute map from sym, compute full array (no hidden legs)
      new_map    = Map.compute(self.sym, map_legs, self.backend)
      full_array = self.backend.einsum(subscript, self._array, new_map.array)
      return full_array



  # --- Transformation methods ---------------------------------------------- #

  def transform(self, path):
      return transform(self, path)



  # --- Properties and get/set methods -------------------------------------- #

  @property
  def array(self):
      return self._array

  @property
  def backend(self):
      return self._backend

  @property
  def sym(self):
      return self._sym

  @property
  def ndim(self):
      return self._ndim

  @property
  def shape(self):

      if  ISNOT(self.sym):
          return self.dense_shape

      return self.sym_shape + self.dense_shape

  @property
  def dense_shape(self):
      return self._dense_shape

  @property
  def sym_shape(self):
      if ISNOT(self.sym):
         return None
      return self.sym.truncated_shape

  def get_dense_dims(self, legs):
      return {leg: dim for leg, dim in zip(legs, self.dense_shape)}

  # ------------------------------------------------------------------------- #





def transform(tensor, path):  

    # Num of transformation nodes
    num_nodes = len(path) 

    # (1) Trivial case: path with no transformations
    if  num_nodes == 0:
        return tensor.as_new_tensor()

    # (2) Construct a list of symlegs and maps for transformation einsum
    symlegs    = []  
    map_arrays = []

    for i, node in enumerate(path): 

        # First node: add start legs of tensor
        if  i == 0:
            symlegs += [node.start_legs]

        # Middle nodes: add map legs and map itself
        symlegs    += [node.map_legs]
        map_arrays += [node.map_array] 

        # Last node: add end legs of tensor
        if  i == num_nodes - 1:
            symlegs += [node.end_legs]

    # (3) Generate dummy dense legs, combine them with symlegs list   
    #     (maps have no dense legs, only initial and final tensors do)
    denselegs = util.make_legs(tensor.ndim)

    legs     = symlegs
    legs[0]  = symlegs[0]  + denselegs
    legs[-1] = symlegs[-1] + denselegs

    # (4) Make contraction subscript from legs list
    subscript = util.legs_to_subscript(*legs)

    # (5) Contract tensor with all the transformation maps
    backend   = tensor.backend
    new_array = backend.einsum(subscript, tensor.array, *map_arrays)
    return tensor.as_new_tensor(new_array)                             
  


















































































































































































































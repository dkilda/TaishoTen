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

      if   ISNOT(sym): 
   
           # No symmetry: treat as a dense tensor
           self._ndim        = array.ndim
           self._dense_shape = array.shape

      else:
           # Get ndim and dense shape of the array
           self._ndim        = sym.num_legs
           self._dense_shape = array.shape[-sym.num_legs : ] 


  @classmethod
  def create(cls, *args, **kwargs):

      # Create a new tensor
      tensor = cls(*args, **kwargs)

      # Check tensor symmetry is consistent with tensor array dims
      sym = tensor.sym
      arr = tensor.array

      if IS(sym):

         # Check ndim of array matches truncated ndim
         msg = "Tensor: invalid ndim of input array" 
         trunc_arr_ndim = sym.num_symlegs + sym.num_legs - 1 

         assertequal(arr.ndim, trunc_arr_ndim, msg)

         # Check array shape is consistent with symmetric shape
         msg = "Tensor.create: array and symmetry truncated shapes must match"

         arr_shape = arr.shape[: sym.num_symlegs - 1]
         sym_shape = sym.truncated_shape
         assertequal(arr_shape, sym_shape, msg)
         
         # Symmetrize tensor
         tensor.symmetrize()

      return tensor



  # --- Auxiliary methods --------------------------------------------------- #

  def as_new(self, array=None, sym=None):

      # Create a new Tensor object
      if  ISNOT(array):
          array = self._array

      if  ISNOT(sym):
          sym = self._sym

      return type(self)(array, sym)



  def _make_legs(self):

      # Make dense and symmetric legs (ndim = num of dense legs)
      denselegs     = util.make_legs(self.ndim)
      symlegs       = util.make_symlegs(denselegs, self.sym.fullsigns) 
      trunc_symlegs = util.truncate(symlegs)

      return denselegs, symlegs, trunc_symlegs      



  def symmetrize(self):

      # Make legs
      denselegs, symlegs, trunc_symlegs = self._make_legs()    
      trunc_legs = trunc_symlegs + denselegs

      # Get map from symmetry
      mp = self.get_map(symlegs) 
  
      # Compute map squared (e.g. AB,AB->A)
      sub = util.legs_to_subscript(symlegs, symlegs, trunc_symlegs) 
      mp_squared_array = self.backend.einsum(sub, mp.array, mp.array)    

      # Compute symmetrized array (e.g. Aab,A->Aab)
      sub = util.legs_to_subscript(trunc_legs, trunc_symlegs, trunc_legs) 
      self._array = self.backend.einsum(sub, self.array, mp_squared_array)
      return self



  def get_full_array(self): 

      # If no symmetry
      if  ISNOT(self.sym):
          return self.array

      # Make legs
      denselegs, symlegs, trunc_symlegs = self._make_legs() 
      
      # Make truncated-to-full subscript (e.g. Aab,AB->ABab)
      full_legs  = symlegs       + denselegs
      trunc_legs = trunc_symlegs + denselegs
      subscript  = util.legs_to_subscript(trunc_legs, symlegs, full_legs) 

      # Get map from symmetry
      mp = self.get_map(symlegs) 

      # Compute full array (no hidden legs)
      full_array = self.backend.einsum(subscript, self.array, mp.array)
      return full_array



  # --- Transformation methods ---------------------------------------------- #

  def transform(self, *args, **kwargs):
      return transform(self, *args, **kwargs)



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

  def get_map(self, map_legs):
      if ISNOT(self.sym):
         return None
      return Map.compute(self.sym, map_legs, self.backend)

  # ------------------------------------------------------------------------- #





def transform(tensor, path, denselegs=None):  

    # Num of transformation nodes
    num_nodes = len(path) 

    # (1) Trivial case: path with no transformations
    if  num_nodes == 0:
        return tensor.as_new()

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

    # (3) Combine dense legs with symmetric legs list   
    #     (maps have no dense legs, only initial and final tensors do)
    if  ISNOT(denselegs):
        denselegs = util.make_legs(tensor.ndim)

    legs     = symlegs
    legs[0]  = symlegs[0]  + denselegs
    legs[-1] = symlegs[-1] + denselegs

    # (4) Make contraction subscript from legs list
    subscript = util.legs_to_subscript(*legs)

    # (5) Contract tensor with all the transformation maps
    backend   = tensor.backend
    new_array = backend.einsum(subscript, tensor.array, *map_arrays)
    return tensor.as_new(new_array)                             
  


def random(shape, sym=None, backend=None, **kwargs):

    backend = backends.get_backend(backend)
    array   = backend.random(shape)

    return Tensor.create(array, sym, backend, **kwargs)
    




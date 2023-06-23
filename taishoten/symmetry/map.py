#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

import numpy as np
import copy  as cp
import itertools

import taishoten.backends as backends
import taishoten.util as util

from taishoten.util import Str
from taishoten.util import assertequal
from taishoten.util import IS, ISNOT, ARE, ARENOT

from .symmetry import set_symtol




def contract_maps(mapA, mapB, out_legs=None, tol=None):

    # Make sure mapA and mapB have the same backend
    msg = "contract_maps: mapA and mapB must have the same backend"
    assertequal(mapA.backend, mapB.backend, msg)
    backend = mapA.backend

    # Default output legs of (mapA, mapB) contraction: 
    # assume legs represented by the same letter are always contracted
    if  ISNOT(out_legs):
        out_legs = (mapA.legs | mapB.legs) - (mapA.legs & mapB.legs)

    # Make the subscript
    subscript = util.legs_to_subscript(mapA.legs, mapB.legs, out_legs)

    # Contract mapA and mapB, 
    # find indices of nonzero entries in the resulting array
    out   = backend.einsum(subscript, mapA.array, mapB.array)
    shape = backend.shape(out)
    idx   = backend.find_nonzeros(out, tol=set_symtol(tol))

    # Compute mapC from these indices and the desired shape of the map
    mapC = Map.compute_from_idx(shape, idx, legs=out_legs, backend=backend) 
    return mapC




class Map:

   def __init__(self, array, legs, backend=None):

       # Set array, legs, and backend of this map
       self._backend = backends.get_backend(backend)
       self._legs    = Str(legs)
       self._array   = self.backend.asarray(array)


   @classmethod
   def compute(cls, sym, legs, backend=None):

       # If symmetry "sym" already has a map, simply return that map
       if  sym.has_map:
           new_map = sym.map.copy() 
           new_map._set_legs(legs)
           return new_map

       # Flatten out symshape by combining all symlegs into one, which gives
       # flat_symlabels[idx] = sgn1*q1[i] + sgn2*q2[j] + sgn3*q3[k] + ..., 
       # where idx = combo(i,j,k,...), 
       # and i = sym index along leg-1, j = sym index along leg-2, etc. 
       #
       # Subtract Qtot from symrange, apply modulus, and sum absolute values
       # of vector (multi-D) symlabels: 
       # flat_symlabels[idx] 
       # --> \sum_n |(flat_symlabels[idx][n] - Qtot[n]) % mod|
       #
       flat_symlabels  = sym.flatten_symlabels()
       flat_symlabels -= sym.qtot
       flat_symlabels  = sym.apply_mod(flat_symlabels)
       flat_symlabels  = sym.sum_abs_inner_symlabels(flat_symlabels)

       # Find all indices where flat_symlabels = 0, i.e. the conservation law 
       # \sum_n |(sgn1*q1[i] + sgn2*q2[j] + ... - Qtot)[n] % mod| = 0 
       # is satisfied.
       idx   = sym.find_zeros(flat_symlabels) 
       shape = sym.shape

       # Create map operator as an array with shape = defined by symlabels,
       # and where array[i,j,k,...] = 1 if flat_symlabels[idx] = 0 
       # (i.e. the conservation law is satisfied and symlabels of all symlegs
       #  in "sym" add up to zero), and array[i,j,k,...] = 1 otherwise.
       new_map = cls.compute_from_idx(shape, idx, legs, backend=backend)

       # Register the new map with the symmetry "sym"
       sym.set_map(new_map)
       return new_map



   @classmethod
   def compute_from_idx(cls, shape, idx, legs, backend=None):

       # Get backend
       backend = backends.get_backend(backend)

       # Generate the map itself: set new_map[i,j,k,...] = 1  
       # at each index = combo(i,j,k,...) where index \in input idx. 
       # These input idx correspond to positions where 
       # flat_symlabels[idx] = 0 (i.e. all symlabels add up to zero).
       vals      = backend.ones(len(idx))
       map_array = backend.zeros(shape)
       map_array = backend.put(map_array, idx, vals)

       # Create new map
       return cls(map_array, legs, backend=backend)



   def copy(self, deep=False):

       # Copy map
       if   deep:
            array = self._backend.copy(self._array) 
       else:
            array = self._array

       return type(self)(array, self._legs, backend=self._backend)



   def contract(self, other, out_legs=None, tol=None):

       # Contract two symmetry maps
       return contract_maps(self, other, out_legs=out_legs, tol=tol)


   # --- Private auxiliary methods ------------------------------------------ #

   def _set_legs(self, legs):

       # Private method: nothing from outside can modify the map 
       # (i.e. once created, map is an immutable object)

       # Assert that number of legs is valid
       msg = "Map._set_legs: number of legs and "\
             "number of dimensions must be equal"
       assertequal(len(legs), self.ndim, msg)

       # Set map legs
       self._legs = Str(legs)
       return self



   # --- Properties --------------------------------------------------------- #

   @property
   def shape(self):
       return self._array.shape

   @property
   def ndim(self):
       return self._array.ndim

   @property
   def legs(self):
       return self._legs

   @property
   def array(self):
       return self._array

   @property
   def backend(self):
       return self._backend





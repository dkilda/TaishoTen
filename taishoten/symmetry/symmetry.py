#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

import numpy as np
import copy  as cp
import itertools

import taishoten.util as util

from taishoten.util import StrSet
from taishoten.util import assertequal, noniterable
from taishoten.util import IS, ISNOT, ARE, ARENOT

from taishoten.params import ALPHABET, SIGN, FLIP, SYMTOL



def flip_signs(signs): 
    return ''.join(FLIP[sgn] for sgn in x)



def get_symlabels_shape(symlabels, truncate=False):

    if  truncate:
        symlabels = symlabels[:-1]

    return (len(s) for s in symlabels)



def sum_vector_symlabels(symlabels_nD):

    symlabels_1D = np.sum(abs(symlabels_nD), axis=-1)
    return symlabels_1D



def find_zeros(symlabels, tol=SYMTOL):

    idx = np.where(abs(symlabels) < tol)[0]
    return idx



def invert_matrix_3D(x): 

    # Define the inverse of a 3x3 matrix
    det = np.linalg.det(x)

    def cofac_row(i): 
        return np.cross(x[i-2], x[i-1])           

    inv_x = [cofac_row(i) / det for i in range(3)]
    inv_x = np.asarray(inv_x).T
    return inv_x



def round_to_nint(array): 
    return np.rint(array)



def round_to_floor(array):
    return np.floor(array)
    


def round_to_float(array, decimal_places=10):
    return np.round_(array, decimal_places)



def mod_3D(symlabels, mod, folding=False):

    symlabels = np.dot(symlabels, invert_matrix_3D(mod))

    if   folding:

         symlabels = round_to_float(symlabels)
         symlabels = symlabels - round_to_floor(symlabels)
         symlabels = round_to_float(symlabels)

    else:
         symlabels = symlabels - round_to_nint(symlabels)

    return symlabels  









class Symmetry:

   def __init__(self, fullsigns, symlabels, qtot=0, mod=None, tol=SYMTOL):

       # Construct symlabels, full signs (including zero-signed legs) 
       # and signs (without zero-signed legs)
       self._symlabels = np.asarray([np.asarray(s) for s in symlabels])  
       self._fullsigns = fullsigns
       self._signs     = ''.join(sgn for sgn in fullsigns if sgn != '0')

       # Assert the number of signs and symlabels is equal 
       # (it is equal to the number of symlegs)
       msg = "Symmetry: number of signs and symlabels must be equal"
       assertequal(len(self.signs), len(symlabels), msg)

       # Set the total symlabel, modulus, and tolerance
       self._qtot = qtot
       self._mod  = mod
       self._tol  = tol
       self._ndim = util.NotImplementedField

       # Map storage (if registered)
       self._map = None



   def __new__(cls, *args, **kwargs):

       # Prevent Symmetry class from a direct instantiation
       if  cls is Symmetry:
           raise TypeError("Symmetry class cannot be instantiated directly")
       return object.__new__(cls, *args, **kwargs)



   # --- Properties: basic -------------------------------------------------- #

   @property 
   def fullsigns(self):
       return self._fullsigns

   @property 
   def signs(self):
       return self._signs

   @property
   def symlabels(self):
       return self._symlabels

   @property
   def qtot(self):
       return self._qtot

   @property
   def mod(self):
       return self._mod

   @property
   def ndim(self):
       return self._ndim



   # --- Properties: shape and signs ---------------------------------------- #

   @property 
   def shape(self):
       return get_symlabels_shape(self.symlabels)
       
   @property
   def shape_truncated(self):
       return get_symlabels_shape(self.symlabels, truncate=True)

   @property
   def flipped_fullsigns(self):
       return flip_signs(self.fullsigns)

   @property
   def flipped_signs(self):
       return flip_signs(self.signs)
       
   @property
   def num_signed_symlegs(self):
       return len(self.signs)

   @property
   def num_unsigned_symlegs(self):
       return self.fullsigns.count('0')



   # --- Map-related methods ------------------------------------------------ #

   @property
   def map(self):
       return self._map

   @property
   def has_map(self):
       return IS(self._map)

   def set_map(self, map_):
       self._map = map_
       return self

   def unset_map(self):
       self._map = None
       return self

       

   # --- Auxiliary methods for manipulating symlabels ----------------------- #

   def apply_mod(self, symlabels):
       raise NotImplementedError("Symmetry: must implement apply_mod method")


   def fold(self, symlabels):
       raise NotImplementedError("Symmetry: must implement fold method")


   def find_zeros(symlabels):
       return find_zeros(symlabels, tol=self._tol)


   def sum_vector_symlabels(symlabels_nD):
       return sum_vector_symlabels(symlabels_nD)



   # --- Computation with symlabels ----------------------------------------- #

   def flatten_symlabels(self, indices=None, phase=1):

       """
       When we merge legs specified by indices = (i,j,k,...), we need to 
       combine their symlabels: flattening out the shape means flattening out 
       the symlabels.

       We obtain flat symlabels, each entry idx = combo(i,j,k,...) is given by
      
            flat_symlabels[idx] = sgn1*q1[i] + sgn2*q2[j] + sgn3*q3[k] + ...
      
       which corresponds to the sum of [symlabels/qnums] along all legs:
 
       i = sym index, q1[i] = symlabel along leg-1,
       j = sym index, q2[j] = symlabel along leg-2, 
       ... etc ...

       To add symlabels[idx] of each leg = idx (idx is the i'th leg out of 
       those being merged), we need to reshape them into the expanded shape: 

       expanded_shape[-1] = ndim of symmetry
       expanded_shape[k]  = dim  of symlabels[idx] if k = i 
                          = 1                      else 

               
       ************************************************************************

       E.g. Consider symlabels q1 = [0,1], q2 = [0,1,2], signstr = "+-":

            expanded_shape_1 = [2,1]
            expanded_shape_2 = [1,3]
            
            flat_symlabels =  (1) * reshape(q1, shape=[2,1]) 
                           + (-1) * reshape(q2, shape=[1,3]) 

                           = array([[0], [1]]) + array([[0,-1,-2]]) 
                           = array([[0,-1,-2], [1,0,1]])

            Flatten out the array, gives all combo symlabels:

            --> flat_symlabels = array([0, -1, -2, 1, 0, 1]), 


       ************************************************************************
      
       """

       # By default, we merge/flatten all symmetric indices
       if  indices is None:
           indices = range(len(self.symlabels))


       # Make expanded shape for symlabels of index = idx, 
       # idx is the i'th index out of those being merged.
       def expanded_shape(i, idx):

           shape    = [1,]*len(indices) + [self.ndim]
           shape[i] = len(self.symlabels[idx])
           return shape


       # Build flattened-out combo symlabels 
       # by merging the given index set "indices"
       flat_symlabels = 0

       for i, idx in enumerate(indices):

           # Reshape symlabels of index = idx into expanded shape,
           # get the sign of index = idx 
           shape         = expanded_shape(i, idx) 
           idx_symlabels = self.symlabels[idx].reshape(shape)
           idx_sign      = self.signs[idx]

           # Accumulate signed symlabels of each index = idx:
           # flat_q += sgn_{idx} * q_{idx} * phase
           flat_symlabels += idx_symlabels * SIGN[idx_sign] * phase 


       # Flatten out the combo symlabels array and return it
       flat_symlabels = flat_symlabels.reshape(-1, self.ndim)
       return flat_symlabels



   def aux_symlabels(self, indices, phase=1):

       # Generate dummy legs corresponding to the full signs of this symmetry,
       # pick out the signed dummy legs
       legs        = StrSet(ALPHABET[ : len(self.fullsigns)])
       signed_legs = util.cut_unsigned(legs, self.fullsigns)  

       # The input "indices" define idx of middle legs. 
       # Get mid_idx = idx of middle signed legs.
       mid_legs = [legs[i] for i in indices]
       mid_idx  = util.gen_idx(mid_legs, signed_legs)
       mid_idx  = list(mid_idx)

       # Get edge idx = idx of non-middle signed legs.
       all_idx  = range(len(signed_legs))
       edge_idx = [i for i in all_idx if i not in mid_idx]

       # Construct flattened symlabels by combining mid legs and edge legs
       # qMID[idx]  = sgnM1 * qM1[i] + sgnM2 * qM2[j] + ..., 
       # qEDGE[idx] = sgnE1 * qE1[i] + sgnE2 * qE2[j] + ..., 
       # where idx = combo(i,j,...), so that we have the conservation law
       # qMID[idx] - qEDGE[jdx] + Qtot = 0
       mid_symlabels   = self.flatten_symlabels(mid_idx,   phase=phase)
       edge_symlabels  = self.flatten_symlabels(edge_idx,  phase=-1*phase)
       edge_symlabels += self.qtot

       # Fold symlabels (apply mod and take unique symlabels only).
       # We'll need unique symlabels only in the next step.
       mid_symlabels  = self.fold(mid_symlabels)
       edge_symlabels = self.fold(edge_symlabels) 

       # Align middle and edge symlabels. The alignment procedure combines 
       # middle and edge symlabels and picks out only those middle 
       # symlabels that satisfy the conservation law 
       # fold(qMID[idx] - qEDGE[jdx] + Qtot) = 0
       # and constructs auxiliary symlabels from them. 
       # 
       # So aux_symlabels = unique middle symlabels that satisfy the above 
       # conservation law (= the minimal set of symlabels needed to 
       # construct the auxiliary bond; aux symlabels must satisfy 
       # the conservation laws at both the node with middle-and-aux-legs 
       # and the node with edge-and-aux legs).
       # 
       aux_symlabels = self.align_symlabels(mid_symlabels, edge_symlabels)
       return aux_symlabels



   def align_symlabels(self, symlabels_11, symlabels_22):

       # Combine two sets of flat symlabels: combo[i,j] = q1[i] - q2[j] 
       # (assuming symlabels_11 has "+" and symlabels_22 has "-" sign)
       #
       # E.g. symlabels_11 = [0,1], symlabels_22 = [0,1,2]
       # --> combo_symlabels = array([[0,1,2], [1,0,1]])
       #
       combo_symlabels = abs(symlabels_11[:, None] - symlabels_22[None, :])

       # Combined flat symlabels w/ scalar elements will have ndim = 2.
       # If ndim > 2, means our symlabels have vector elements.
       if  combo_symlabels.ndim > 2:
           combo_symlabels = self.sum_vector_symlabels(combo_symlabels)

       # Find all indices along axis=0 (corresponding to symlabels_11)
       # where combo symlabels are zero, i.e. the conservation law 
       # combo[i,j] = q1[i] - q2[j] = 0 is satisfied. 
       idx = self.find_zeros(combo_symlabels)

       # The indices "idx" point to symlabels from symlabels_11 that satisfy 
       # the conservation law: use them to make a new set of aligned symlabels.
       # These aligned symlabels satisfy the conservation laws with both
       # symlabels_11 and symlabels_22.
       aligned_symlabels_11 = np.array([symlabels_11[i] for i in idx])
       return aligned_symlabels_11


   # ------------------------------------------------------------------------ #





class Symmetry1D(Symmetry):

   def __init__(self, *args, **kwargs):

       # Base class initialization
       super().__init__(*args, **kwargs)

       # Make sure all symlabels are scalar
       msg = "Symmetry1D: all symlabel elements must be scalar"
       assert all(noniterable(elem) for symlab in self.symlabels \
                                    for elem   in symlab), msg 
 
       # Set the symmetry dim
       self._ndim = 1


   # --- Auxiliary methods for manipulating symlabels ----------------------- #

   def apply_mod(self, symlabels):
       
       # If mod is None: nothing to do
       if  self.mod is None:
           return symlabels

       # Apply mod to symlabels 
       symlabels = np.mod(symlabels, self.mod)
       return symlabels


   def fold(self, symlabels):
       
       # Fold flat symlabels: apply mod and return sorted unique symlabels.
       # Gives the minimal representation of our symlabel set.
       symlabels = self.apply_mod(symlabels)
       symlabels = np.unique(symlabels)
       return symlabels

   # ------------------------------------------------------------------------ #






class Symmetry3D(Symmetry):

   def __init__(self, *args, **kwargs):

       # Base class initialization
       super().__init__(*args, **kwargs)

       # Set the symmetry dim
       self._ndim = 3

       # Make sure symlabel elements are 3D
       msg = "Symmetry3D: symlabel element and symmetry dimensions must match"
       assertequal(self.ndim, len(self.symlabels[0][0]), msg)

       # Make sure all symlabel elements have the same size
       msg = "Symmetry3D: all symlabel elements must have the same size"
       assert all(len(elem) == self.ndim for symlab in self.symlabels \
                                         for elem   in symlab), msg

       # Make sure the modulus dimension is 3D
       if  self.mod is not None:
           msg = "Symmetry3D: dimensions of modulus and symmetry must match"
           assertequal(self.ndim, len(self.mod), msg)

       
   # --- Auxiliary methods for manipulating symlabels ----------------------- #

   def apply_mod(self, symlabels, folding=False):
       
       # If mod is None: nothing to do
       if  self.mod is None:
           return symlabels

       # Apply mod to symlabels (multiply symlabels by inverted mod matrix)
       symlabels = mod_3D(symlabels, self.mod, folding=folding)
       return symlabels


   def fold(self, symlabels):
       
       # Fold flat symlabels: apply mod and return sorted unique symlabels.
       # Gives the minimal representation of our symlabel set.
       #
       # NB for ndim > 1, symlabel elements are vectors: 
       #    we assume flat symlabels with shape = (len(symlabels), ndim) 
       #    and get unique symlabel vectors along axis = 0.
       #
       symlabels = self.apply_mod(symlabels, folding=True)
       symlabels = np.unique(symlabels, axis=0)
       return symlabels

   # ------------------------------------------------------------------------ #

















































































































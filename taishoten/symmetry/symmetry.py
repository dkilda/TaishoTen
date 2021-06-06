#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

import numpy as np
import copy  as cp
import itertools

import taishoten.util as util

from taishoten.util import Str
from taishoten.util import assertequal, noniterable
from taishoten.util import IS, ISNOT, ARE, ARENOT


# --- Auxiliary functions --------------------------------------------------- #

def set_symtol(tol=None):

    if ISNOT(tol):
       tol = 2**(-16)

    return tol



def get_symlabels_shape(symlabels, truncate=False):

    if  truncate:
        symlabels = symlabels[:-1]

    return (len(s) for s in symlabels)



def sum_abs_inner_symlabels(symlabels):

    symlabels = abs(symlabels)

    if  symlabels.ndim > 1:
        symlabels = np.sum(symlabels, axis=-1)

    return symlabels



def find_zeros(symlabels, tol=None):

    tol = set_symtol(tol)
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



def sum_meshgrid(*xs):

    # Make sure the inner element dimension 
    # for all vectors x \in xs is the same
    def xshape(x): return x.shape[1:]

    xdim = list(xshape(xs[0]) 
    assert all(xshape(x) == xdim for x in xs), \
           "sum_meshgrid: all xs must have the same inner dimension"

    # Compute summed meshgrid
    summed_grid = 0

    for i, x in enumerate(xs):

        # Grid shape for vector x \in xs
        shape    = [1,]*len(xs) + xdim
        shape[i] = len(x)
        shape    = tuple(shape)

        # Accumulate reshaped vectors to form a summed grid
        summed_grid = summed_grid + x.reshape(shape)

    # Corner case
    if  summed_grid == 0:
        summed_grid = np.array([])

    return summed_grid
        



def flatten(x, xdim=None):

    # Flatten a vector, except for its inner elements of dimension "xdim"
    if  ISNOT(xdim):
        return x.reshape((-1,))

    return x.reshape((-1, xdim))
    



# --- Symmetry base class --------------------------------------------------- #

class Symmetry:

   def __init__(self, fullsigns, symlabels, qtot=0, mod=None, tol=None):

       # Construct symlabels, full signs (including zero-signed legs) 
       # and signs (without zero-signed legs)
       self._symlabels = [np.asarray(s) for s in symlabels]  
       self._fullsigns = fullsigns
       self._signs     = ''.join(sgn for sgn in fullsigns if sgn != '0')

       # Assert the number of signs and symlabels is equal 
       # (it is equal to the number of symlegs)
       msg = "Symmetry: number of signs and symlabels must be equal"
       assertequal(len(self.signs), len(symlabels), msg)

       # Set the total symlabel, modulus, and tolerance
       self._qtot = qtot
       self._mod  = mod
       self._tol  = set_symtol(tol=tol)
       self._ndim = None 

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

   @property
   def num_legs(self):
       return len(self.fullsigns)

   @property
   def num_symlegs(self):
       return len(self.signs)



   # --- Properties: shape and signs ---------------------------------------- #

   @property 
   def shape(self):
       return get_symlabels_shape(self.symlabels)
       
   @property
   def truncated_shape(self):
       return get_symlabels_shape(self.symlabels, truncate=True)

   @property
   def flipped_fullsigns(self):
       return flip_signs(self.fullsigns)

   @property
   def flipped_signs(self):
       return flip_signs(self.signs)
       


   # --- Map-related methods ------------------------------------------------ #

   @property
   def map(self):
       return self._map

   @property
   def has_map(self):
       return IS(self._map)

   def set_map(self, map_):

       msg = "Symmetry.set_map: map and symmetry shapes must be compatible"
       assertequal(map_.shape, self.shape, msg)
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


   def find_zeros(self, symlabels):
       return find_zeros(symlabels, tol=self._tol)


   def sum_abs_inner_symlabels(self, symlabels):
       return sum_abs_inner_symlabels(symlabels)



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



       # Default indices of vectors that we'll use to construct meshgrid
       if  ISNOT(indices):
           indices = range(self.num_symlegs)

       # Corner case: no indices to flatten
       if  len(indices) == 0:
           raise ValueError("Symmetry.flatten_symlabels: "\
                            "number of indices cannot be zero")

       # Compute signed symlabels for given indices: 
       # Q_i = q_i * sgn_i * phase, i \in indices  
       symlabels = [self.symlabels[i] * self.signs[i] * phases \ 
                                                        for i in indices]
       # Accumulate signed symlabels 
       # Q = \sum_i Q_i of all indices i \in indices, flatten them out 
       summed_symlabels = sum_meshgrid(*symlabels)
       flat_symlabels   = flatten(summed_symlabels, self.ndim)
       return flat_symlabels



   def aux_symlabels(self, indices, phase=1): 

       # The "indices" specifies legs to be put on LHS of conservation law 
       # equation. We convert "indices" from all-indices to signed-indices 
       # and group symlegs into LHS and RHS.
       all_idx   = range(self.num_symlegs)
       left_idx  = util.cut_unsigned(indices, self.fullsigns) 
       right_idx = [i for i in all_idx if i not in right_idx]     

       # Construct flattened symlabels for LHS and RHS legs: 
       # qLHS[idx] =   sgnL[0] * qL[0,i] + sgnL[1] * qL[1,j] + ...
       # qRHS[idx] = - sgnR[0] * qR[0,i] - sgnR[1] * qR[1,j] - ... + Qtot
       # where idx = combo(i,j,...).
       left_symlabels   = self.flatten_symlabels(left_idx,  phase=phase)
       right_symlabels  = self.flatten_symlabels(right_idx, phase=-1*phase)
       right_symlabels += self.qtot

       # Fold symlabels (apply mod and take unique symlabels only)
       # We'll only need unique symlabels in the next step.
       left_symlabels  = self.fold(left_symlabels)
       right_symlabels = self.fold(right_symlabels) 

       # Align LHS symlabels with RHS symlabels: combine LHS and RHS symlabels 
       # and pick only those LHS symlabels that satisfy the conservation law 
       # fold(qLHS[idx] - qRHS[jdx]) = 0, 
       # and construct auxiliary symlabels from them. 
       # 
       # So aux_symlabels = unique LHS symlabels that satisfy the above 
       # conservation law (= the minimal set of LHS symlabels needed to 
       # construct the auxiliary bond; aux symlabels must satisfy 
       # the conservation laws at both the node with LHS-and-aux-legs 
       # and the node with RHS-and-aux legs).
       # 
       aux_symlabels = self.align_symlabels(left_symlabels, right_symlabels)
       return aux_symlabels
       


   def align_symlabels(self, lhs, rhs):

       # Combine two sets of flat symlabels: combo[i,j] = q1[i] - q2[j] 
       # (assuming symlabels_11 has "+" and symlabels_22 has "-" sign)
       #
       # E.g. symlabels_11 = [0,1], symlabels_22 = [0,1,2]
       # --> combo_symlabels = array([[0,1,2], [1,0,1]])
       #
       merged = sum_meshgrid(lhs, -rhs)
       merged = abs(merged)

       # Combined flat symlabels w/ scalar elements will have ndim = 2.
       # If ndim > 2, means our symlabels have vector elements.
       if  merged.ndim > 2:
           merged = self.sum_abs_inner_symlabels(merged)

       # Find all indices along axis=0 (corresponding to symlabels_11)
       # where combo symlabels are zero, i.e. the conservation law 
       # combo[i,j] = q1[i] - q2[j] = 0 is satisfied. 
       idx = self.find_zeros(merged)

       # The indices "idx" point to symlabels from symlabels_11 that satisfy 
       # the conservation law: use them to make a new set of aligned symlabels.
       # These aligned symlabels satisfy the conservation laws with both
       # symlabels_11 and symlabels_22.
       aligned_lhs = lhs[list(idx)]
       return aligned_lhs




# --- 1D Symmetry class ----------------------------------------------------- #

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




# --- 3D Symmetry class ----------------------------------------------------- #

class Symmetry3D(Symmetry):

   def __init__(self, *args, **kwargs):

       # Base class initialization
       super().__init__(*args, **kwargs)

       # Set the symmetry dim
       self._ndim = 3

       # Sanity checks: make sure symmetry is consistent with 3D geometry. 
       if  IS(self.mod):
           msg = "Symmetry3D: dimensions of modulus and symmetry must match"
           assertequal(self.ndim, len(self.mod), msg)

       msg = "Symmetry3D: symlabel element and symmetry dimensions must match"
       assert all(len(elem) == self.ndim \
                  for symlb in self.symlabels for elem  in symlb), msg


       
   # --- Auxiliary methods for manipulating symlabels ----------------------- #

   def apply_mod(self, symlabels, folding=False):
       
       # If mod is None: nothing to do
       if  ISNOT(self.mod):
           return symlabels

       # Apply mod matrix to symlabels: 
       # r = S % G -> r = S*G^-1 - round_to_int(S*G^-1)
       #
       # Scaled symlabels = S*G^-1 \in (-1,1)
       # Using r = S/G - rint(S/G) gives signed scaled remainder.
       #
       # So apply_mod finds signed remainder and does not unscale symlabels.
       # It should be used if we only care about finding zeros S % G = 0.
       #
       # Otherwise, if one intends to use symlabels for other purposes, 
       # fold() should be used instead. 
       # 
       #
       symlabels = np.dot(symlabels, invert_matrix_3D(self.mod))
       symlabels = symlabels - np.rint(symlabels)
       return symlabels



   def fold(self, symlabels):
       
       # Fold flat symlabels: apply mod and return sorted unique symlabels.
       # Gives the minimal representation of our symlabel set.
       #
       # NB for ndim > 1, symlabel elements are vectors: 
       #    we assume flat symlabels with shape = (len(symlabels), ndim) 
       #    and get unique symlabel vectors along axis = 0.
       #

       # Auxiliary functions 
       # (including the number of decimal places for rounding)
       def rounded(x):
           return np.round_(x, 10) 

       def floor(x):
           return np.floor(rounded(x))

       def unique(x):
           return np.unique(rounded(x), axis=0)

       # If no modulus
       if  ISNOT(self.mod):
           return unique(symlabels)
          
       # Apply mod matrix to symlabels, pick only unique ones 
       # r = S % G -> r = G*[S*G^-1 - floor(S*G^-1)]
       #
       # First, we find unsigned remainder. 
       # Because scaled symlabels = S*G^-1 \in (-1,1), applying floor(S*G^-1) 
       # means all +ve components -> 0, all -ve components -> -1.
       # So S*G^-1 - floor(S*G^-1) folds -ve side onto the +ve side. In the 
       # language of original symlabels S \in (-G,G), it means S -> S + G for 
       # all -ve components of S. Thus, from now on, all components are +ve.   
       # 
       # Next, we pick out the unique symlabels 
       # (with -ve components folded onto +ve side).
       #
       # Finally, we unscale symlabels: r -> r = G*r.
       # 
       symlabels = np.dot(symlabels, invert_matrix_3D(self.mod))
       symlabels = symlabels - floor(symlabels)
       symlabels = unique(symlabels)
       symlabels = np.dot(symlabels, self.mod)
       return symlabels






























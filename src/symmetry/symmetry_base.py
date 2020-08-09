#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

import numpy as np
import copy  as cp
import itertools

from util import Str
from util import assert_equal, isiterable, del_from_list, multisorted
from util import NotImplementedField
from util import subscript_to_legs, legs_to_subscript
from util import symlegs_and_denselegs_to_subscript

from params import SIGN, FLIP, ALPHABET


class Symmetry:

  def __init__(self, signs, symrange, qtot=0, mod=None):

      # Sign string, sym pattern, qtot, modulus
      self._signs     = signs
      self._puresigns = self._make_puresigns(signs)
      self._symrange  = self._make_symrange(symrange)  
      self._qtot      = qtot
      self._mod       = mod

      # Symmetry dim
      self._ndim = NotImplementedField


  def __new__(cls, signs, symrange, qtot=0, mod=None, *args, **kwargs):

      # Prevent Symmetry from instantiation
      if  cls is Symmetry:
          raise TypeError("Symmetry class cannot be instantiated")
      return object.__new__(cls, *args, **kwargs)



  # --- Properties ------------------------------------------------------------------------------------------

  @property
  def signs(self):
      return cp.deepcopy(self._signs)

  @property
  def puresigns(self):
      return cp.deepcopy(self._puresigns)

  @property
  def symrange(self):
      return cp.deepcopy(self._symrange)

  @property
  def qtot(self):
      return cp.deepcopy(self._qtot)

  @property
  def mod(self):
      return cp.deepcopy(self._mod)

  @property
  def ndim(self):
      return cp.deepcopy(self._ndim)



  # --- Get methods -----------------------------------------------------------------------------------------

  def get_flipped_signs(self):
      return self._get_flipped_signs(self._signs)


  def get_flipped_puresigns(self):
      return self._get_flipped_signs(self._puresigns)


  def get_num_signed_legs(self):
      return len(self._puresigns)


  def get_num_unsigned_legs(self):
      return self._signs.count('0')


  def get_full_shape(self, shape):

      # Get full dense + symmetric legs shape
    
      # Make sure sign string and dense shape have equal num of legs
      assert_equal(len(self._signs), len(shape), \
                   "Symmetry.get_full_shape: sign string and shape must have equal num of legs")

      # Get sym and dense shape
      symlegs_shape   = [len(srng) for srng in self._symrange[:-1]]
      denselegs_shape = list(shape)

      # Combine sym and dense shape to get the full shape
      full_shape = symlegs_shape + denselegs_shape 
      full_shape = tuple(full_shape)
      return full_shape


  def get_symrange_shape(self, symrange=None):

      # Get shape of the symrange
      if  symrange is None:
          symrange = self._symrange

      shape = [len(srng) for srng in symrange]
      return shape



  # --- Auxiliary methods for construction ------------------------------------------------------------------

  def _make_puresigns(self, signs):

      # Find indices of unsigned (sgn = '0') legs
      indices = [i for i, sgn in enumerate(signs) if sgn = '0']

      # Delete them from the sign string
      puresigns = del_from_list(signs, indices)

      # Ensure that symrange and puresigns have equal lengths
      assert_equal(len(puresigns), len(self._symrange), \
                   "Symmetry: symrange and puresigns must have equal lengths")
      return puresigns


  @staticmethod
  def _get_flipped_signs(signs):

      # Flip signs in the sign string
      flipped_signs = [FLIP[sgn] for sgn in signs]
      flipped_signs = ''.join(flipped_signs)

      return flipped_signs


  @staticmethod
  def _make_symrange(symrange):
      raise NotImplementedError("Symmetry: must implement _make_symrange() method")



  # --- Symrange computation --------------------------------------------------------------------------------

  def make_flat_symrange(self, indices=None, phase=1):

      # In the resulting combo symrange, each entry (i,j,k) is given by:
      #
      #      flat_symrange[i,j,k] = sgn1*q1[i] + sgn2*q2[j] + sgn3*q3[k]
      #
      # and corresponds to the sum of [symlabels/qnums]: 
      #     label-i along leg-1, label-j along leg-2, label-k along leg-3, etc.
      # 
      # E.g. symranges n1 = [0,1], n2 = [0,1,2], signstr = "+-"
      #      flat_symrange = (1) * reshape(n1, shape=[2,1]) + (-1) * reshape(n2, shape=[1,3]) =
      #                    = array([[0], [1]]) + array([[0,-1,-2]]) =
      #                    = array([[0,-1,-2], [1,0,1]])
      # Flatten out:
      #      flat_symrange = array([0, -1, -2, 1, 0, 1]), contains all combo symlabels
      #

      # Default set of indices 
      if  indices is None:
          indices = range(len(self.symrange))

      # Num of symmetric legs included in this [combo symrange]
      num_sym_indices = len(indices)

      # Build a flattened-out combo symrange: 
      #       an [array] of [combo symmetry labels along a given set of legs]
      # To do this, we loop over all legs, get their symranges and add them together.
      flat_symrange = 0

      for cnt, i in enumerate(indices):

          # Construct an effective shape for the symrange of leg-i, 
          #                                      that allows addition of different symranges. 
          # For leg-i, shape[i] = 1 at each i, except shape[i] = num of symlabels along leg-i.
          shape      = [1,] * num_sym_indices + [self.ndim]
          shape[cnt] = len(self.symrange[i])

          # Each [symrange of leg-i] = [a list of symmetry labels along leg-i]
          # (1) get symrange corresponding to leg-i, 
          # (2) reshape it into an [effective shape] 
          # (3) multiply all sym labels by the [sign of leg-i] and by the [sign phase factor]
          # (4) add symrange to [flat symrange]
          flat_symrange += self.symrange[i].reshape(shape) * SIGN[self.puresigns[i]] * phase

      # Flatten out the combo symrange and return it
      flat_symrange = flat_symrange.reshape(-1, self.ndim)
      return flat_symrange


  def make_aux_symrange(self, indices, phase=1):

      # Generate a legstring of equal length to signstring: purify from '0' legs
      legs     = Str(ALPHABET[ : len(self.signs)])
      purelegs = legs.purify(self.signs)

      # Get contracted (CON) legs, omitting any 0-legs
      CON_idx = [purelegs.find(leg) for i, leg in enumerate(legs) if self.signs[i] != '0']

      # Get resulting (RES) legs    
      RES_idx = [i for i in range(len(purelegs)) if i not in CON_idx]

      # Construct symranges from CON legs and from RES legs, compute
      # --> qCON[idx]  =   sgnC1 * qC1[i] + sgnC2 * qC2[j] + ...,        idx = combo(i,j,...) 
      # --> qREST[idx] = - sgnR1 * qR1[i] - sgnR2 * qR2[j] - ... + Qtot, idx = combo(i,j,...)
      # So that we have qCON[idx] - qRES[jdx] = 0  
      CON_symrange  = self.make_flat_symrange(indices=CON_idx, phase=phase)
      RES_symrange  = self.make_flat_symrange(indices=RES_idx, phase=-1*phase)
      RES_symrange += self.qtot

      # Fold CON and RES symranges: apply mod and return only the unique symlabels 
      # (in the next step, we'll pick out the symlabels that are in principle able 
      #  to satisfy the conservation laws between CON and RES legs, so only unique symlabels will be needed)
      CON_symrange = self.fold(CON_symrange)
      RES_symrange = self.fold(RES_symrange)

      # Merge [combo CON symrange] and [combo REST symrange] to obtain [auxiliary symrange]:
      # symlabel conservation must hold between CON and RES leg groups, i.e. qCON[i] - qREST[j] = 0.
      # AUX symrange contains "qCON[i]" symlabels of CON symrange that are able to satisfy the conservation law above. 
      AUX_symrange = self.merge(CON_symrange, RES_symrange)
      return AUX_symrange


  def merge(self, symrange_A, symrange_B):

      # Combine two sets of symlabels (assuming symrange A has "+", symrange B has "-" sign): 
      # --> combo_symrange[i,j] = q1[i] - q2[j]
      # 
      # E.g. symrange_A = [0,1], symrange_B = [0,1,2] 
      # --> combo_symrange = array([[0,1,2], [1,0,1]])
      #
      combo_symrange = abs(symrange_A[:, None] - symrange_B)

      # Sum out symrange elements if they're iterable 
      # (i.e. if more than 2 axes are present, the last axis runs over each element)
      if  combo_symrange.ndim > 2:
          combo_symrange = self.sum_over_elements(combo_symrange)

      # Find all indices where combo_symrange is zero, 
      #      i.e. the symlabel conservation law is satisfied, q1[i] - q2[j] = 0 and idx = (i,j)
      #      i.e. idx[0] = index i of symrange A, idx[1] = index j of symrange B
      #
      # Take [indices i of symrange A] = idx[0] that are able to satisfy the conservation laws
      idx = self.find_zeros(combo_symrange) 

      # Get [symlabels of symrange A] that are able to satisfy the conservation laws: 
      #                               use them to form a new, smaller symrange
      merged_symrange = np.array([symrange_A[i] for i in idx])
      return merged_symrange



  # --- Auxiliary methods for symrange computation ----------------------------------------------------------

  def apply_mod(self, symrange):
      raise NotImplementedError("Symmetry: must implement apply_mod() method")


  def fold(self, symrange):
      raise NotImplementedError("Symmetry: must implement fold() method")


  def sum_over_elements(self, symrange):

      # Sum over each symrange element
      summed_symrange = np.sum(abs(symrange), axis=-1)
      return summed_symrange


  def find_zeros(self, symrange):

      # Find indices correpsonding to zero irrep labels in symrange
      idx = np.where(symrange < self._tol)[0]
      return idx

  # ---------------------------------------------------------------------------------------------------------


















































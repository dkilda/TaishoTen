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

from .symmetry import SymFactory



class SymCon:

  def __init__(self, symA, symB, denselegs):

      # Sanity check: the moduli of A and B must be either None or equal
      if  symA.mod is not None and symB.mod is not None:
          assert np.allclose(symA.mod, symB.mod), "SymCon: A and B must have equal moduli"

      # Input: symmetries and dense legs
      self._symA      = symA
      self._symB      = symB
      self._denselegs = denselegs
      self._symlegs   = None

      # Make symmetry legs of A and B
      self._make_symlegs("A", symA.signs)
      self._make_symlegs("B", symB.signs)

      # Get contraction legs, and the relative phase of A and B
      self._CON   = self._symlegs["A"] & self._symlegs["B"]
      self._phase = self._compute_relative_phase()

      # Output symmetry
      self._symC = None

      # Backends
      assert_equal(self._symA.backend, self._symB.backend, \
                   "SymCon: symA and symB must have the same backend")


  def __new__(cls, symA, symB, denselegs, *args, **kwargs):

      # Prevent SymCon and SymConWithIrrepMaps from instantiation
      if  cls in (SymCon, SymConWithIrrepMaps):
          raise TypeError("SymCon and SymConWithIrrepMaps classes cannot be instantiated")
      return object.__new__(cls, *args, **kwargs)



  # --- Properties ------------------------------------------------------------------------------------------

  @property
  def symA(self):
      return self._symA

  @property
  def symB(self):
      return self._symB

  @property
  def denselegs(self):
      return self._denselegs

  @property
  def symlegs(self):
      return self._symlegs

  @property
  def CON(self):
      return self._CON

  @property
  def phase(self):
      return self._phase

  @property
  def ndim(self):
      return self._symA.ndim

  @property
  def backend(self):
      return self._symA.backend



  # --- Get methods -----------------------------------------------------------------------------------------

  def get_denselegs(self):
      return cp.deepcopy(self._denselegs)      

  def get_symlegs(self):
      return cp.deepcopy(self._symlegs)

  def get_output_symmetry(self):
      return cp.deepcopy(self._symC)

  def get_phase(self):
      return cp.deepcopy(self._phase)

  def get_contracted_legs(self):
      return cp.deepcopy(self._CON) 



  # --- Auxiliary methods for construction ------------------------------------------------------------------

  def _make_symlegs(self, key, signs):

      if  self._symlegs is None:
          self._symlegs = {}

      self._symlegs[key] = self._denselegs[key].purify(signs)
      return symlegs


  def _compute_relative_phase(self):

      # Shortcut
      puresigns_A = self.symA.puresigns
      puresigns_B = self.symB.puresigns
      CON         = self.CON

      # If we've no CON legs, phase is trivially 1
      if  len(self.CON) == 0:
          phase = 1
          return phase 

      # Loop over all CON legs, determine their phases
      # -- if both CON legs have the same sign, then phase = -1, else phase = 1 
      #
      phaselist = [(-1 if sgnA == sgnB else 1) \
                       for sgnA, sgnB, leg in self.symdata_of_legs(puresigns_A, puresigns_B, CON)]

      # All CON legs must have the same phase (only 1 unique element allowed):
      # else the symmetry is invalid
      if  np.unique(phaselist).size != 1:
          raise ValueError("SymCon._compute_relative_phase(): Invalid symmetry")

      # Set phase
      phase = phaselist[0]
      return phase



  # --- Symrange computation --------------------------------------------------------------------------------

  def merge(self, symrange_A, symrange_B):

      # Get moduli of symA, symB
      mod_A = self.symB.mod
      mod_B = self.symA.mod

      # Use internal merge method from either of the symmetries
      _merge = self.symA.merge

      if   mod_A is None and mod_B is None:

           # Merge symranges A and B directly (no moduli)
           return _merge(symrange_A, symrange_B)

      elif mod_A is not None and mod_B is not None:

           # Merge symranges A and B directly (make sure their moduli are equal)
           assert np.allclose(mod_A, mod_B), "SymCon.merge: must have mod_A = mod_B"
           return _merge(symrange_A, symrange_B)

      elif mod_A is not None and mod_B is None:

           # Fold A using modulus of B 
           # (to put A on the same footing as B, so we can merge them) 
           folded_symrange_A = self.symB.fold(symrange_A)
           return _merge(symrange_B, folded_symrange_A)

      else:
           # Fold B using modulus of A 
           # (to put B on the same footing as A, so we can merge them)
           folded_symrange_B = self.symA.fold(symrange_B)
           return _merge(symrange_A, folded_symrange_B)



  # --- Compute contracted output symmetry ------------------------------------------------------------------

  def compute_output_sym(self):
      
      # Compute output mod, qtot, signs string
      mod   = self._compute_output_mod()
      qtot  = self._compute_output_qtot()
      signs = self._compute_output_signs()

      # Make symlegs for C (remove 0-legs, make uppercase) using the new signs 
      # (must do this before computing the new symrange)
      self._make_symlegs("C", signs) 
 
      # Compute output symrange
      symrange = self._compute_output_symrange()

      # Create output symmetry
      self._symC = SymFactory(signs, symrange, qtot, mod, backend=self.backend)
      return self


  def _compute_output_signs(self):

      # Signs (if phase = -1, flip all signs in [the sign-string of tensor A])
      signs_A   = self.symA.signs if self.phase == 1 else self.symA.get_flipped_signs()
      signs_B   = self.symB.signs
      denselegs = self.denselegs

      # Compute output signs
      signs = self._compute_output_symdata(signs_A, signs_B, self.denselegs)
      signs = ''.join(signs)
      return signs


  def _compute_output_symrange(self):

      # Compute output symrange
      symrange = self._compute_output_symdata(self._symrange_A, self._symrange_B, self._symlegs)
      return symrange


  def _compute_output_qtot(self):

      # Compute output Qtot
      qtot = (self.phase * self.symA.qtot) + self.symB.qtot
      return qtot


  def _compute_output_mod(self):

      # Shortcut function
      isnone = lambda x: x is None

      # Determine the combined modulus
      if   isnone(self.symA) and     isnone(self.symB):
           mod = None

      elif isnone(self.symA) and not isnone(self.symB):
           mod = self.symB.mod

      elif isnone(self.symB) and not isnone(self.symA):
           mod = self.symA.mod

      else:
           assert np.allclose(self.symA.mod, self.symB.mod), \
                  "compute_output_mod: moduli must be equal"
           mod = self.symA.mod

      return mod


  def _compute_output_symdata(self, symdata_A, symdata_B, legs):

      # Loop over all legs of C, determine the symdata of tensor C
      # If leg is in A, get its symdata from symdata of A, else get it from B
      # (any non-zero leg of C must be present in either A or B)    
      #
      symdata_C = [(sA if leg in legs["A"] else sB) \
                       for sA, sB, leg in self.symdata_of_legs(symdata_A, symdata_B, legs)]

      return symdata_C



  # --- Auxiliary methods for output symmetry ---------------------------------------------------------------

  @staticmethod
  def symdata_of_legs(datA, datB, legs): 
 
      # Make an iterator over symmetry data corresponding to different legs, 
      # e.g. signs or symranges of different legs

      for leg in legs["C"]:

          # Position of [leg from C] in A, B legs
          iA = legs["A"].find(leg)
          iB = legs["B"].find(leg)  

          # Get sym info from A, B
          leg_datA = datA[iA]
          leg_datB = datB[iB]

          yield leg_datA, leg_datB, leg

  # ---------------------------------------------------------------------------------------------------------


























































#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools

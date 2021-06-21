#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import copy  as cp
import numpy as np

import lib
import util

import taishoten as tn
import taishoten.symmetry as tnsym
from taishoten import Str







class TestSymmetryContraction3D:

   @pytest.fixture(autouse=True)
   def request_symmetries(self, fixt_symmetries_3D):
       self._symmetries_and_data = fixt_symmetries_3D


   @pytest.fixture(autouse=True)
   def request_symmetry_contractions(self, fixt_symcon_3D):
       self._symcon_and_data = fixt_symcon_3D


   def symmetries_and_data(self, key):
       sym, data, legs, symlegs = self._symmetries_and_data[key]
       return sym, data, legs, symlegs


   def symmetry_contractions_and_data(self, key):
       symcon, sym, legs, symlegs = self._symcon_and_data[key]
       return symcon, sym, legs, symlegs



   # --- Test constructor --------------------------------------------------- #

   @pytest.mark.parametrize("key, phase",                          \
   [                                                               \
   [("ijk,++-,M1",       "klm,++-,M1",     "ijlm,+++-,M1"),    1], \
   [("ijlm,+++-,M1",     "nojl,++--,M1",   "inom,+++-,M1"),    1], \
   [("inom,+++-,M1",     "inp,++-,M1",     "pom,--+,M1"),     -1], \
   [("ixnyom,+0+0+-,M1", "inzxp,++00-,M1", "pyzom,-00-+,M1"), -1], \
   ])
   def test_construct(self, key, phase):

       out, sym, legs, symlegs = self.symmetry_contractions_and_data(key)

       sym1     = cp.deepcopy(sym)
       symlegs1 = cp.deepcopy(symlegs)

       sym1["C"]     = None
       symlegs1["C"] = None

       util.assert_symmetry_contraction(out, sym1, legs, symlegs1, phase)



   @pytest.mark.parametrize("keyA, keyB, legsC",      \
   [                                                  \
   ["ijk,++-,M1", "klm,++-,Invalid,1D", Str("ijlm")], \
   ["ijk,++-,M1", "klm,++-,M2",         Str("ijlm")], \
   ])
   @pytest.mark.xfail
   def test_construct_failed(self, keyA, keyB, legsC):

       symA, _, legsA, _ = self.symmetries_and_data(keyA)
       symB, _, legsB, _ = self.symmetries_and_data(keyB)

       legs   = tn.dictriplet(legsA, legsB, legsC) 
       symcon = tn.SymmetryContraction(symA, symB, legs)



   # --- Test compute() ----------------------------------------------------- #

   @pytest.mark.parametrize("key, phase",                          \
   [                                                               \
   [("ijk,++-,M1",       "klm,++-,M1",     "ijlm,+++-,M1"),    1], \
   [("ijlm,+++-,M1",     "nojl,++--,M1",   "inom,+++-,M1"),    1], \
   [("inom,+++-,M1",     "inp,++-,M1",     "pom,--+,M1"),     -1], \
   [("ixnyom,+0+0+-,M1", "inzxp,++00-,M1", "pyzom,-00-+,M1"), -1], \
   ])
   def test_compute(self, key, phase):

       out, sym, legs, symlegs = self.symmetry_contractions_and_data(key)

       out.compute()
       util.assert_symmetry_contraction(out, sym, legs, symlegs, phase)






































































































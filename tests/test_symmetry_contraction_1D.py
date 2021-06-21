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




class TestSymmetryContraction:


   @pytest.fixture(autouse=True)
   def request_symlabels(self, fixt_symlabels_1D):
       self._symlabels = fixt_symlabels_1D


   @pytest.fixture(autouse=True)
   def request_symmetries(self, fixt_symmetries_1D):
       self._symmetries_and_data = fixt_symmetries_1D


   @pytest.fixture(autouse=True)
   def request_symmetry_contractions(self, fixt_symcon_1D):
       self._symcon_and_data = fixt_symcon_1D


   def symlabels(self, symlegs=None):
       if  symlegs is None:
           return self._symlabels
       return [self._symlabels[l] for l in symlegs]


   def symmetries_and_data(self, key):
       sym, data, legs, symlegs = self._symmetries_and_data[key]
       return sym, data, legs, symlegs


   def symmetries(self, key):
       sym = self._symmetries_and_data[key][0]
       return sym


   def symmetry_contractions_and_data(self, key):
       symcon, sym, legs, symlegs = self._symcon_and_data[key]
       return symcon, sym, legs, symlegs


   def symmetry_contractions(self, key):
       symcon = self._symcon_and_data[key][0]
       return symcon


   # --- Test constructor --------------------------------------------------- #

   @pytest.mark.parametrize("key, phase",                         \
   [                                                              \
   [("ijk,++-",       "klm,++-",       "ijlm,+++-"),        1],   \
   [("ijlm,+++-",     "nojl,++--",     "inom,+++-"),        1],   \
   [("inom,+++-",     "inp,++-",       "pom,--+"),         -1],   \
   [("ixnyom,+0+0+-", "inzxp,++00-",   "pyzom,-00-+"),     -1],   \
   [("ijk,++-,Q1,M3", "klm,++-,Q2,M3", "ijlm,+++-,Q3,M3"),  1],   \
   [("ijk,++-,M3",    "klm,++-",       "ijlm,+++-,M3"),     1],   \
   [("ijk,++-",       "klm,++-,M3",    "ijlm,+++-,M3"),     1],   \
   ])
   def test_construct(self, key, phase):

       out, sym, legs, symlegs = self.symmetry_contractions_and_data(key)

       sym1     = cp.deepcopy(sym)
       symlegs1 = cp.deepcopy(symlegs)

       sym1["C"]     = None
       symlegs1["C"] = None

       util.assert_symmetry_contraction(out, sym1, legs, symlegs1, phase)



   @pytest.mark.parametrize("keyA, keyB, legsC",   \
   [["inom,+++-", "inp,+--", Str("pom")]])
   @pytest.mark.xfail
   def test_construct_failed(self, keyA, keyB, legsC):

       symA, _, legsA, _ = self.symmetries_and_data(keyA)
       symB, _, legsB, _ = self.symmetries_and_data(keyB)

       legs   = tn.dictriplet(legsA, legsB, legsC) 
       symcon = tn.SymmetryContraction(symA, symB, legs)




   # --- Test compute() ----------------------------------------------------- #

   @pytest.mark.parametrize("key, phase",                         \
   [                                                              \
   [("ijk,++-",       "klm,++-",       "ijlm,+++-"),        1],   \
   [("ijlm,+++-",     "nojl,++--",     "inom,+++-"),        1],   \
   [("inom,+++-",     "inp,++-",       "pom,--+"),         -1],   \
   [("ixnyom,+0+0+-", "inzxp,++00-",   "pyzom,-00-+"),     -1],   \
   [("ijk,++-,Q1,M3", "klm,++-,Q2,M3", "ijlm,+++-,Q3,M3"),  1],   \
   [("ijk,++-,Q1,M3", "klm,-+-,Q2,M3", "ijlm,--+-,Q1,M3"), -1],   \
   [("ijk,++-,M2",    "klm,++-",       "ijlm,+++-,M2"),     1],   \
   [("ijk,++-",       "klm,++-,M5",    "ijlm,+++-,M5"),     1],   \
   ])
   def test_compute(self, key, phase):

       out, sym, legs, symlegs = self.symmetry_contractions_and_data(key)

       out.compute()
       util.assert_symmetry_contraction(out, sym, legs, symlegs, phase)




   # --- Test output qtot, mod, symlabels, fullsigns ------------------------ #

   @pytest.mark.parametrize("key",                        \
   [                                                      \
   ("ijk,++-",       "klm,++-",       "ijlm,+++-"),       \
   ("ijk,++-,Q1,M3", "klm,++-,Q2,M3", "ijlm,+++-,Q3,M3"), \
   ("ijk,++-,Q1,M3", "klm,-+-,Q2,M3", "ijlm,--+-,Q1,M3"), \
   ])
   def test_output_qtot(self, key):

       symcon, sym, _, _ = self.symmetry_contractions_and_data(key)

       out = symcon.output_qtot()
       ans = sym["C"].qtot

       util.assert_array_close(out, ans)




   @pytest.mark.parametrize("key",                        \
   [                                                      \
   ("ijk,++-",       "klm,++-",       "ijlm,+++-"),       \
   ("ijk,++-,Q1,M3", "klm,++-,Q2,M3", "ijlm,+++-,Q3,M3"), \
   ("ijk,++-,M2",    "klm,++-",       "ijlm,+++-,M2"),    \
   ("ijk,++-",       "klm,++-,M5",    "ijlm,+++-,M5"),    \
   ])
   def test_output_mod(self, key):

       symcon, sym, _, _ = self.symmetry_contractions_and_data(key)

       out = symcon.output_mod()
       ans = sym["C"].mod

       if   ans is None:
            assert out is None
       else:
            util.assert_array_close(out, ans)




   @pytest.mark.parametrize("keyA, keyB, legsC", \
   [                                             \
   ["ijk,++-,M3", "klm,++-,M5", Str("ijlm")],    \
   ])
   @pytest.mark.xfail
   def test_output_mod_failed(self, keyA, keyB, legsC):

       symA, _, legsA, _ = self.symmetries_and_data(keyA)
       symB, _, legsB, _ = self.symmetries_and_data(keyB)

       legs   = tn.dictriplet(legsA, legsB, legsC) 
       symcon = tn.SymmetryContraction(symA, symB, legs)

       out = symcon.output_mod()




   @pytest.mark.parametrize("key",                        \
   [                                                      \
   ("ijk,++-",       "klm,++-",       "ijlm,+++-"),       \
   ("ijlm,+++-",     "nojl,++--",     "inom,+++-"),       \
   ("inom,+++-",     "inp,++-",       "pom,--+"),         \
   ("ixnyom,+0+0+-", "inzxp,++00-",   "pyzom,-00-+"),     \
   ("ijk,++-,Q1,M3", "klm,++-,Q2,M3", "ijlm,+++-,Q3,M3"), \
   ])
   def test_output_symlabels(self, key):

       symcon, sym, _, _ = self.symmetry_contractions_and_data(key)

       out = symcon.output_symlabels(sym["C"].fullsigns)
       ans = sym["C"].symlabels

       util.assert_array_list_close(out, ans)




   @pytest.mark.parametrize("key",                        \
   [                                                      \
   ("ijk,++-",       "klm,++-",       "ijlm,+++-"),       \
   ("ijlm,+++-",     "nojl,++--",     "inom,+++-"),       \
   ("inom,+++-",     "inp,++-",       "pom,--+"),         \
   ("ixnyom,+0+0+-", "inzxp,++00-",   "pyzom,-00-+"),     \
   ("ijk,++-,Q1,M3", "klm,++-,Q2,M3", "ijlm,+++-,Q3,M3"), \
   ])
   def test_output_fullsigns(self, key):

       symcon, sym, _, _ = self.symmetry_contractions_and_data(key)

       out = symcon.output_fullsigns()
       ans = sym["C"].fullsigns

       util.assert_array_list_equal(out, ans)




   # --- Test align_symlabels() --------------------------------------------- #

   @pytest.mark.parametrize("key",                           \
   [                                                         \
   (1, "ijk,++-",       "klm,++-",       "ijlm,+++-"),       \
   (1, "ijk,++-,Q1,M3", "klm,++-,Q2,M3", "ijlm,+++-,Q3,M3"), \
   (1, "ijk,++-",       "klm,++-,M3",    "ijlm,+++-,M3"),    \
   (1, "ijk,++-,M3",    "klm,++-",       "ijlm,+++-,M3"),    \
   ])
   def test_align_symlabels(self, fixt_aligned_symlabels, key):

       symlabelsA, symlabelsB, ans, ans1 = fixt_aligned_symlabels[key]

       symcon = self.symmetry_contractions(key[1:])
       out    = symcon.align_symlabels(symlabelsA, symlabelsB)

       util.assert_array_close(out, ans)
       util.assert_array_close(out, ans1)


 

   # --- Test auxiliary symmetry -------------------------------------------- #

   @pytest.mark.parametrize("key, signs, symlegs, mod",                    \
   [                                                                       \
   [("inom,+++-",     "inp,++-",      "pom,--+"),      "++-", "in", None], \
   [("ixnyom,+0+0+-", "inzxp,++00-",  "pyzom,-00-+"),  "++-", "in", None], \
   [("ijlm,+++-",     "nojl,++--",    "inom,+++-"),    "++-", "jl", None], \
   [("ijlm,+++-,M2",  "nojl,++--",    "inom,+++-,M2"), "++-", "jl", 2   ], \
   [("ijlm,+++-",     "nojl,++--,M2", "inom,+++-,M2"), "++-", "jl", 2   ], \
   ])
   def test_compute_aux_symmetry(self, fixt_aux_symlabels, \
                                       key, signs, symlegs, mod):

       symcon = self.symmetry_contractions(key)
       out = tnsym.symmetry_contraction.compute_aux_symmetry(symcon) 

       symlabels = self.symlabels(symlegs)
       symlabels = [*symlabels, fixt_aux_symlabels[key]]
   
       util.assert_symmetry(out, signs, symlabels, mod=mod)




   # --- Test map contractions ---------------------------------------------- #

   def test_compute_maps(self, fixt_maps):

       key, maps, legs, _, shapes = fixt_maps

       symcon = self.symmetry_contractions(key)
       out    = tnsym.compute_maps(symcon)
       out    = sorted(out, key=lambda x: x.legs)

       util.assert_list(out, maps, fun=util.assert_map_equal)

       for i in range(len(out)):
           util.assert_map(out[i], legs[i], shapes[i])



   def test_compute_pairwise_contractions_of_maps(self, fixt_random_maps):

       init_maps, maps, legs, shapes = fixt_random_maps

       def compute_pairwise_contractions_of_maps(x):
           mod = tnsym.symmetry_contraction
           fun = mod.compute_pairwise_contractions_of_maps
           return fun(x)

       out = compute_pairwise_contractions_of_maps(init_maps)
       out = sorted(out, key=lambda x: x.legs)

       util.assert_list(out, maps, fun=util.assert_map_equal)

       for i in range(len(out)):
           util.assert_map(out[i], legs[i], shapes[i])







































































































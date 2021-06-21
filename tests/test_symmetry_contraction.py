#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import copy  as cp
import numpy as np
import helper_lib as lib

import util
from util import isiterable, noniterable

import taishoten as tn
from taishoten import Str



def np_einsum_of_maps(subscript, legsC, mapA, mapB):

    arrayC = np.einsum(subscript, mapA.array, mapB.array)
    idx    = lib.find_nonzeros(arrayC)
    arrayC = lib.map_from_idx(arrayC.shape, idx, val=1.0)
    out    = tn.Map(arrayC, legsC)
    return out





def make_aux_symlabels(sym, symindices, phase=1):

    signs     = [[], []]
    symlabels = [[], []]
    qtot      = sym.qtot
    mod       = sym.mod

    for k in (0,1):
      for i in symindices[k]:
          signs[k].append(sym.signs[i])
          symlabels[k].append(sym.symlabels[i])

    for k in (0,1):
        signs[k] = ''.join(signs[k])

    aux_symlabels = lib.make_aux_symlabels(signs, symlabels, \
                                           qtot, mod, phase=phase)

    return aux_symlabels




def make_symmetry_1D(fullsigns, symlabels, qtot=0, mod=None, signs=None):

    sym  = tn.Symmetry1D(fullsigns, symlabels, qtot, mod)

    data = {"fullsigns": fullsigns, "signs": signs, \
            "symlabels": symlabels, "qtot": qtot, "mod": mod}

    return sym, data




@pytest.fixture
def symlabels1D():

    Si = np.arange(0,5) 
    Sj = np.arange(0,2)
    Sk = np.arange(1,5)
    Sl = np.arange(0,4)
    Sm = np.arange(1,5)
    Sn = np.arange(0,5)
    So = np.arange(0,2)
    Sp = np.arange(0,5)

    symlabels = [Si,Sj,Sk,Sl,Sm,Sn,So,Sp]
    symlabels = dict(zip("ijklmnop", symlabels))
    return symlabels




@pytest.fixture
def symmetries1D(symlabels1D):
    
    dct = {}
    def make(legs, fullsigns, suffix=None, **kwargs):

        symlabels = [symlabels1D[l] for l in legs if l in symlabels1D]
        symlegs   = ''.join(l       for l in legs if l in symlabels1D)
        symlegs   = symlegs.upper()

        sym, data = make_symmetry_1D(fullsigns, symlabels, **kwargs)

        key = ",".join([legs, fullsigns]) 
        if  suffix:
            key = ",".join([key, suffix]) 

        dct[key]  = (sym, data, Str(legs), Str(symlegs))


    make("ijk", "++-")
    make("ijk", "++-", suffix="M2",            mod=2)
    make("ijk", "++-", suffix="M3",            mod=3)
    make("ijk", "++-", suffix="Q1,M3", qtot=1, mod=3)

    make("klm", "++-")
    make("klm", "++-", suffix="M3",            mod=3)
    make("klm", "++-", suffix="M5",            mod=5)
    make("klm", "++-", suffix="Q2,M3", qtot=2, mod=3)
    make("klm", "-+-", suffix="Q2,M3", qtot=2, mod=3)

    make("ijlm", "+++-")
    make("ijlm", "+++-", suffix="M2",            mod=2)
    make("ijlm", "+++-", suffix="M3",            mod=3)
    make("ijlm", "+++-", suffix="M5",            mod=5)
    make("ijlm", "+++-", suffix="Q3,M3", qtot=3, mod=3)
    make("ijlm", "--+-", suffix="Q1,M3", qtot=1, mod=3)

    make("inp",   "+--")
    make("inp",   "++-")
    make("inzxp", "++00-", signs="++-")

    make("pom",   "--+")
    make("pyzom", "-00-+", signs="--+")

    make("nojl",   "++--")
    make("nojl",   "++--", suffix="M2", mod=2)

    make("inom",   "+++-")
    make("inom",   "+++-", suffix="M2", mod=2)
    make("ixnyom", "+0+0+-", signs="+++-")

    return dct




@pytest.fixture
def symmetry_contractions(symmetries1D):

    dct = {}
    def make(keyA, keyB, keyC):

        symA, _, legsA, symlegsA = symmetries1D[keyA] 
        symB, _, legsB, symlegsB = symmetries1D[keyB]
        symC, _, legsC, symlegsC = symmetries1D[keyC]

        sym     = tn.dictriplet(symA,     symB,     symC)
        legs    = tn.dictriplet(legsA,    legsB,    legsC) 
        symlegs = tn.dictriplet(symlegsA, symlegsB, symlegsC) 

        symcon = tn.SymmetryContraction(symA, symB, legs)
        dct[(keyA, keyB, keyC)] = (symcon, sym, legs, symlegs)


    make("ijk,++-",       "klm,++-",       "ijlm,+++-")
    make("inom,+++-",     "inp,++-",       "pom,--+")
    make("ixnyom,+0+0+-", "inzxp,++00-",   "pyzom,-00-+")

    make("ijk,++-,Q1,M3", "klm,++-,Q2,M3", "ijlm,+++-,Q3,M3")
    make("ijk,++-,Q1,M3", "klm,-+-,Q2,M3", "ijlm,--+-,Q1,M3")
    make("ijk,++-,M3",    "klm,++-",       "ijlm,+++-,M3")
    make("ijk,++-",       "klm,++-,M3",    "ijlm,+++-,M3")
    make("ijk,++-,M2",    "klm,++-",       "ijlm,+++-,M2")
    make("ijk,++-",       "klm,++-,M5",    "ijlm,+++-,M5")

    make("ijlm,+++-",     "nojl,++--",     "inom,+++-")
    make("ijlm,+++-,M2",  "nojl,++--",     "inom,+++-,M2")
    make("ijlm,+++-",     "nojl,++--,M2",  "inom,+++-,M2")

    return dct




@pytest.fixture
def aux_symlabels(symmetry_contractions):   

    dct = {}
    def make(*key):

        symcon, sym, _, _ = symmetry_contractions[key]

        phase = -symcon.phase
        symA  = sym["A"]
        symB  = sym["B"]

        def _make(symindicesA, symindicesB):
              
            Saux_A = make_aux_symlabels(symA, symindicesA)
            Saux_B = make_aux_symlabels(symB, symindicesB, phase=phase)
            Saux   = lib.align_symlabels(Saux_A, Saux_B)

            dct[key] = Saux

        return _make

    symidxA = [(0,1), (2,3)]
    symidxB = [(0,1), (2,)]

    make("inom,+++-",     "inp,++-",      "pom,--+"    )(symidxA, symidxB)
    make("ixnyom,+0+0+-", "inzxp,++00-",  "pyzom,-00-+")(symidxA, symidxB)

    symidxA = [(1,2), (0,3)]
    symidxB = [(2,3), (0,1)]

    make("ijlm,+++-",      "nojl,++--",    "inom,+++-"   )(symidxA, symidxB)
    make("ijlm,+++-,M2",   "nojl,++--",    "inom,+++-,M2")(symidxA, symidxB)
    make("ijlm,+++-",      "nojl,++--,M2", "inom,+++-,M2")(symidxA, symidxB)

    return dct






@pytest.fixture
def aligned_symlabels(symmetry_contractions):  

    dct = {}
    def make(idx, *key):

        def _make(A, B, ans, ans1):
            dct[(idx, *key)] = (A, B, ans, ans1)

        return _make

    A = np.arange(0,5)
    B = np.arange(0,2)
    C = np.arange(1,5)

    ans  = np.array([0,1])
    ans1 = lib.align_symlabels(A, B)
    make(1, "ijk,++-",    "klm,++-",    "ijlm,+++-")(A, B, ans, ans1)

    ans  = np.array([1,2,3,4])
    ans1 = lib.align_symlabels(A, C)
    make(1, "ijk,++-,Q1,M3", "klm,++-,Q2,M3", \
                             "ijlm,+++-,Q3,M3")(A, C, ans, ans1)

    ans  = np.array([1,2])
    ans1 = lib.align_symlabels(C, lib.fold_1D(A, 3))
    make(1, "ijk,++-",    "klm,++-,M3", "ijlm,+++-,M3")(A, C, ans, ans1)

    ans  = np.array([0,1,2])
    ans1 = lib.align_symlabels(A, lib.fold_1D(C, 3))
    make(1, "ijk,++-,M3", "klm,++-",    "ijlm,+++-,M3")(A, C, ans, ans1)

    return dct





@pytest.fixture
def maps_fixt(symmetry_contractions, symlabels1D, aux_symlabels):

    key = ("ijlm,+++-", "nojl,++--", "inom,+++-")

    # Get symmetries
    symcon, sym, legs, symlegs = symmetry_contractions[key]

    Saux      = aux_symlabels[key]
    symlabels = [symlabels1D["j"], symlabels1D["l"], Saux]
    aux_sym   = tn.Symmetry1D("++-", symlabels)

    # Compute initial maps
    mapA = tn.Map.compute(sym["A"], Str("IJLM")) 
    mapB = tn.Map.compute(sym["B"], Str("NOJL"))
    mapQ = tn.Map.compute(aux_sym,  Str("JLQ"))
  
    # Compute map contractions
    maps, legs, shapes = make_map_contractions(mapA, mapB, mapQ, Saux)
    return key, maps, legs, shapes




@pytest.fixture
def random_maps_fixt(aux_symlabels):

    # Auxiliary symlabels (for shape)
    Saux = aux_symlabels[("ijlm,+++-", "nojl,++--", "inom,+++-")]

    # Compute initial random maps
    mapA = lib.create_random_map(Str("IJLM"), (5,2,4,4),       14) 
    mapB = lib.create_random_map(Str("NOJL"), (5,2,2,4),       10)
    mapQ = lib.create_random_map(Str("JLQ"),  (2,4,len(Saux)), 10)

    initial_maps = [mapA, mapB, mapQ]

    # Compute map contractions
    maps, legs, shapes = make_map_contractions(mapA, mapB, mapQ, Saux)

    return initial_maps, maps, legs, shapes





def make_map_contractions(mapA, mapB, mapQ, Saux):

    # Compute map contractions
    mapAB = np_einsum_of_maps("IJLM,NOJL->IMNO", Str("IMNO"), mapA, mapB)
    mapAQ = np_einsum_of_maps("IJLM,JLQ->IMQ",   Str("IMQ"),  mapA, mapQ)
    mapBQ = np_einsum_of_maps("NOJL,JLQ->NOQ",   Str("NOQ"),  mapB, mapQ)

    maps = [mapA, mapB, mapQ, mapAB, mapAQ, mapBQ]
    maps = sorted(maps, key=lambda x: x.legs)

    legs   = [Str("IJLM"), \
              Str("IMNO"), \
              Str("IMQ"),  \
              Str("JLQ"),  \
              Str("NOJL"), \
              Str("NOQ")]

    shapes = [(5,2,4,4),       \
              (5,4,5,2),       \
              (5,4,len(Saux)), \
              (2,4,len(Saux)), \
              (5,2,2,4),       \
              (5,2,len(Saux))]

    return maps, legs, shapes






    

class TestSymmetryContraction:


   @pytest.fixture(autouse=True)
   def request_symlabels(self, symlabels1D):
       self._symlabels = symlabels1D


   @pytest.fixture(autouse=True)
   def request_symmetries(self, symmetries1D):
       self._symmetries_and_data = symmetries1D


   @pytest.fixture(autouse=True)
   def request_symmetry_contractions(self, symmetry_contractions):
       self._symcon_and_data = symmetry_contractions


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
   def test_align_symlabels(self, aligned_symlabels, key):

       symlabelsA, symlabelsB, ans, ans1 = aligned_symlabels[key]

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
   def test_compute_aux_symmetry(self, aux_symlabels, \
                                       key, signs, symlegs, mod):

       symcon = self.symmetry_contractions(key)
       out = tn.symmetry.symmetry_contraction.compute_aux_symmetry(symcon) 

       symlabels = self.symlabels(symlegs)
       symlabels = [*symlabels, aux_symlabels[key]]
   
       util.assert_symmetry(out, signs, symlabels, mod=mod)




   # --- Test map contractions ---------------------------------------------- #

   def test_compute_maps(self, maps_fixt):

       key, maps, legs, shapes = maps_fixt

       symcon = self.symmetry_contractions(key)
       out    = tn.symmetry.compute_maps(symcon)
       out    = sorted(out, key=lambda x: x.legs)

       util.assert_list(out, maps, fun=util.assert_map_equal)

       for i in range(len(out)):
           util.assert_map(out[i], legs[i], shapes[i])



   def test_compute_pairwise_contractions_of_maps(self, random_maps_fixt):

       init_maps, maps, legs, shapes = random_maps_fixt

       def compute_pairwise_contractions_of_maps(x):
           mod = tn.symmetry.symmetry_contraction
           fun = mod.compute_pairwise_contractions_of_maps
           return fun(x)

       out = compute_pairwise_contractions_of_maps(init_maps)
       out = sorted(out, key=lambda x: x.legs)

       util.assert_list(out, maps, fun=util.assert_map_equal)

       for i in range(len(out)):
           util.assert_map(out[i], legs[i], shapes[i])







































































































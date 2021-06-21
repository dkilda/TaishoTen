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



def make_symmetry_3D(fullsigns, symlabels, qtot=0, mod=None, signs=None):

    sym  = tn.Symmetry3D(fullsigns, symlabels, qtot, mod)

    data = {"fullsigns": fullsigns, "signs": signs, \
            "symlabels": symlabels, "qtot":  qtot,  "mod": mod, "ndim": 3}

    return sym, data




@pytest.fixture
def mod3D():
    mod  = (2 * np.pi / 5) * np.eye(3)
    return mod




@pytest.fixture
def symlabels3D(mod3D):

    dct = {}
    def make(mod3D, key):
        dct[key] = lib.make_symlabels_3D(mod3D, key)

    make(mod3D, (3,3,1))
    make(mod3D, (2,2,1))

    return dct




@pytest.fixture
def symmetries3D(symlabels3D, mod3D):

    dct = {}
    def make(key):
        def _make(*args, **kwargs):  
            dct[key] = make_symmetry_3D(*args, **kwargs)
        return _make

    kpts = symlabels3D[(3,3,1)]
    mod  = mod3D

    make("A1")("+--",    [kpts]*3)
    make("A2")("+--",    [kpts]*3,               mod=mod)
    make("B1")("++--",   [kpts]*4,               mod=mod)
    make("B2")("+0+--",  [kpts]*4,               mod=mod, signs="++--")
    make("C1")("--+-",   [kpts]*4, qtot=kpts[2], mod=mod)
    make("C2")("--0+0-", [kpts]*4, qtot=kpts[2], mod=mod, signs="--+-")

    kpts = symlabels3D[(2,2,1)]

    make("D1")("+--",    [kpts]*3)
    make("D2")("+--",    [kpts]*3,               mod=mod)
    make("E1")("--+-",   [kpts]*4, qtot=kpts[2], mod=mod)
    make("E2")("--0+0-", [kpts]*4,               mod=mod, signs="--+-")

    return dct





class TestSymmetry3D:


   @pytest.fixture(autouse=True)
   def request_mod(self, mod3D):
       self.mod = mod3D



   @pytest.fixture(autouse=True)
   def request_symlabels(self, symlabels3D):
       self.symlabels = symlabels3D



   @pytest.fixture(autouse=True)
   def request_symmetries_and_data(self, symmetries3D):
       self.symmetries_and_data = symmetries3D



   def symmetries(self, key):
       sym, _ = self.symmetries_and_data[key]
       return sym




   @pytest.mark.parametrize("key", ["A1", "A2", \
                                    "B1", "B2", \
                                    "C1", "C2", \
                                   ])
   def test_construct(self, key):

       sym, data = self.symmetries_and_data[key] 
       util.assert_symmetry(sym, **data)




   @pytest.mark.parametrize("which_symlabels, which_mod", \
                             [[0,1], [1,0], [1,1]])
   @pytest.mark.xfail
   def test_construct_failed(self, which_symlabels, which_mod):

       wrong_mod  = np.ones((2,3))
       wrong_kpts = lib.randn(9,2) 

       mod  = (self.mod,                wrong_mod)[which_mod]
       kpts = (self.symlabels[(3,3,1)], wrong_kpts)[which_symlabels]

       sym = tn.Symmetry3D("+--", [kpts]*3, mod=mod)




   @pytest.mark.parametrize("key, shape", [["A1", (9,9,9)],   \
                                           ["C1", (9,9,9,9)], \
                                           ["C2", (9,9,9,9)], \
                                           ["D1", (4,4,4)],   \
                                           ["E1", (4,4,4,4)], \
                                           ["E2", (4,4,4,4)], \
                                          ])
   def test_shape(self, key, shape):

       sym = self.symmetries(key) 
       assert sym.shape           == shape
       assert sym.truncated_shape == shape[:-1]




   @pytest.mark.parametrize("key", [(2,2,1), (3,3,1)])
   def test_sum_abs_inner_symlabels(self, key):

       sym  = self.symmetries("A2")
       kpts = self.symlabels[key]

       out = sym.sum_abs_inner_symlabels(kpts)
       ans = np.sum(abs(kpts), axis=-1)
 
       assert type(out) is np.ndarray
       assert all(noniterable(v) for v in out)
       assert all(v >= 0         for v in out)
       util.assert_array_close(out, ans)




   @pytest.mark.parametrize("key, indices, phase", \
                            [
                             ["A2",  [0,1,2],   1], \
                             ["C1",  [0,1,2,3], 1], \
                             ["C1",  [1],       1], \
                             ["C1",  [0,2],     1], \
                             ["C1",  [0,2,3],   1], \
                             ["C1",  [2,0,3],   1], \
                             ["C1",  [2,0,3],  -1], \
                             ["C2",  [0,1,2,3], 1], \
                             ["C2",  [1,0],     1], \
                             ["C2",  [1,2],    -1], \
                             ["C2",  [3,1,2],  -1], \
                            ]) 
   def test_flatten_symlabels(self, key, indices, phase):

       sym = self.symmetries(key)

       signs     = ''.join([sym.signs[i]     for i in indices]) 
       symlabels =         [sym.symlabels[i] for i in indices] 

       out  = sym.flatten_symlabels(indices, phase)
       ans  = lib.flatten_symlabels(signs, symlabels, phase, ndim=3)

       assert type(out) is np.ndarray
       assert out.shape == (lib.sym_size(symlabels), 3)

       util.assert_array_close(out, ans)



   """
   @pytest.mark.parametrize("key, indices, symindices, phase", \
                            [
                             ["A1",  [0,2],    ([0,2],   [1]),    1], \
                             ["A2",  [0,2],    ([0,2],   [1]),    1], \
                             ["A2",  [0,2],    ([0,2],   [1]),   -1], \
                             ["C1",  [0,3],    ([0,3],   [1,2]),  1], \
                             ["C1",  [3,0],    ([3,0],   [1,2]),  1], \
                             ["C2",  [0,5],    ([0,3],   [1,2]),  1], \
                             ["C2",  [0,5,3],  ([0,3,2],   [1]),  1], \
                            ]) 
   def test_aux_symlabels(self, key, indices, symindices, phase):

       sym = self.symmetries(key) 

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

       out = sym.aux_symlabels(indices, phase)     
       ans = lib.make_aux_symlabels(signs, symlabels, qtot, mod, phase)

       assert type(out) is np.ndarray
       util.assert_array_close(out, ans)
   """



   @pytest.mark.parametrize("key, symindices, phase", \
                            [
                             ["A1",  ([0,2],   [1]),    1], \
                             ["A2",  ([0,2],   [1]),    1], \
                             ["A2",  ([0,2],   [1]),   -1], \
                             ["C1",  ([0,3],   [1,2]),  1], \
                             ["C1",  ([3,0],   [1,2]),  1], \
                             ["C2",  ([0,3],   [1,2]),  1], \
                             ["C2",  ([0,3,2], [1]),    1], \
                            ]) 
   def test_aux_symlabels(self, key, symindices, phase):

       sym = self.symmetries(key) 

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

       out = sym.aux_symlabels(symindices[0], phase)     
       ans = lib.make_aux_symlabels(signs, symlabels, qtot, mod, phase)

       assert type(out) is np.ndarray
       util.assert_array_close(out, ans)




   @pytest.mark.parametrize("key, indices, which", [["A1", [0,2], (3,3,1)]])
   def test_aux_symlabels_by_hand(self, key, indices, which):

       sym = self.symmetries(key)

       ans = self.symlabels[which]
       out = sym.aux_symlabels(indices)

       assert type(out) is np.ndarray
       util.assert_array_close(out, ans)




   def test_apply_mod(self):
    
       # Input
       scaled_symlabels \
             = (1/3) * np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                                 [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                                 [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])

       symlabels = self.mod[0,0] * scaled_symlabels

       # Expected result
       ans = (1/3) * np.array([[ 0, 0, 0], [ 0, 1, 0], [ 0, 1, 0], \
                               [-1, 0, 0], [ 1,-1, 0], [ 1,-1, 0], \
                               [-1, 0, 0], [ 1, 1, 0], [-1, 1, 0]])

       # Test
       sym = self.symmetries("A2")
       out = sym.apply_mod(symlabels)
       util.assert_array_close(out, ans)




   def test_fold(self):
    
       # Input
       scaled_symlabels \
             = (1/3) * np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                                 [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                                 [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])

       symlabels = self.mod[0,0] * scaled_symlabels

       # Expected result
       ans = (1/3) * np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], 
                               [1, 2, 0], [2, 0, 0], [2, 1, 0]])
       ans = self.mod[0,0] * ans
       ans = np.round_(ans, 10)

       # Test
       sym = self.symmetries("A2")
       out = sym.fold(symlabels)
       util.assert_array_close(out, ans)






class TestSymmetryUtil:

   def test_sum_meshgrid(self, symlabels3D):

       kpts      = symlabels3D[(3,3,1)]
       symlabels = [kpts]*4

       out  = tn.symmetry.symmetry.sum_meshgrid(*symlabels)
       ans  = lib.sum_meshgrid(*symlabels)

       util.assert_array_close(out, ans)



   def test_flatten(self, symlabels3D):

       kpts      = symlabels3D[(3,3,1)]
       symlabels = [kpts]*4

       summed_symlabels = lib.sum_meshgrid(*symlabels)

       out  = tn.symmetry.symmetry.flatten(summed_symlabels)
       ans  = lib.flatten(summed_symlabels)

       util.assert_array_close(out, ans)



   def test_get_symlabels_shape(self):

       symlabels = np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                             [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                             [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])

       out = tn.symmetry.symmetry.get_symlabels_shape([symlabels]*4)
       assert out == (9,9,9,9)



   def test_sum_abs_inner_symlabels(self): 

       symlabels = np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                             [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                             [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])

       ans = np.array([0,1,2,1,2,3,2,3,4])
       out = tn.symmetry.symmetry.sum_abs_inner_symlabels(symlabels)
       util.assert_array_equal(out, ans)










@pytest.fixture
def symmetries3D_1(symlabels3D, mod3D):

    kpts = symlabels3D[(3,3,1)]
    mod  = mod3D 
 
    dct = {}
    def make(legs, fullsigns, suffix=None, **kwargs):

        symlegs = ''.join(l for i,l in enumerate(legs) if fullsigns[i] != '0')
        symlegs = symlegs.upper()
        symlabels = [kpts]*len(symlegs)  

        sym, data = make_symmetry_3D(fullsigns, symlabels, **kwargs)

        key = ",".join([legs, fullsigns]) 
        if  suffix:
            key = ",".join([key, suffix]) 

        dct[key] = (sym, data, Str(legs), Str(symlegs))

    make("ijlm", "+++-", suffix="M1", mod=mod)
    make("nojl", "++--", suffix="M1", mod=mod)
    make("ijk",  "++-",  suffix="M1", mod=mod)
    make("klm",  "++-",  suffix="M1", mod=mod)
    make("klm",  "++-",  suffix="M2", mod=2*mod)

    make("inom", "+++-",  suffix="M1", mod=mod)
    make("inp",   "++-",  suffix="M1", mod=mod)
    make("pom",   "--+",  suffix="M1", mod=mod)

    make("ixnyom", "+0+0+-", suffix="M1", mod=mod, signs="+++-")
    make("inzxp",  "++00-",  suffix="M1", mod=mod, signs="++-")
    make("pyzom",  "-00-+",  suffix="M1", mod=mod, signs="--+")

    sym = tn.Symmetry1D("++-", [np.arange(0,2)]*3)
    dct["klm,++-,Invalid,1D"] = (sym, [], Str("klm"), Str("KLM"))

    return dct





@pytest.fixture
def symmetry_contractions(symmetries3D_1):

    dct = {}
    def make(keyA, keyB, keyC):

        symA, _, legsA, symlegsA = symmetries3D_1[keyA] 
        symB, _, legsB, symlegsB = symmetries3D_1[keyB]
        symC, _, legsC, symlegsC = symmetries3D_1[keyC]

        sym     = tn.dictriplet(symA,     symB,     symC)
        legs    = tn.dictriplet(legsA,    legsB,    legsC) 
        symlegs = tn.dictriplet(symlegsA, symlegsB, symlegsC) 

        symcon = tn.SymmetryContraction(symA, symB, legs)
        dct[(keyA, keyB, keyC)] = (symcon, sym, legs, symlegs)

    make("ijk,++-,M1",       "klm,++-,M1",     "ijlm,+++-,M1")
    make("ijlm,+++-,M1",     "nojl,++--,M1",   "inom,+++-,M1")
    make("inom,+++-,M1",     "inp,++-,M1",     "pom,--+,M1")
    make("ixnyom,+0+0+-,M1", "inzxp,++00-,M1", "pyzom,-00-+,M1")

    return dct





class TestSymmetryContraction3D:

   @pytest.fixture(autouse=True)
   def request_symmetries(self, symmetries3D_1):
       self._symmetries_and_data = symmetries3D_1


   @pytest.fixture(autouse=True)
   def request_symmetry_contractions(self, symmetry_contractions):
       self._symcon_and_data = symmetry_contractions


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







'''
'''



































































































































































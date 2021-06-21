#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import copy  as cp
import numpy as np

import lib
import util
from lib import isiterable, noniterable

import taishoten as tn
import taishoten.symmetry as tnsym
from taishoten import Str






@pytest.fixture
def fixt_symmetries_3D(fixt_symlabels_3D, fixt_mod_3D):

    dct = {}
    def make(key):
        def _make(*args, **kwargs):  
            dct[key] = lib.make_symmetry_3D(*args, **kwargs)
        return _make

    kpts = fixt_symlabels_3D[(3,3,1)]
    mod  = fixt_mod_3D

    make("A1")("+--",    [kpts]*3)
    make("A2")("+--",    [kpts]*3,               mod=mod)
    make("B1")("++--",   [kpts]*4,               mod=mod)
    make("B2")("+0+--",  [kpts]*4,               mod=mod, signs="++--")
    make("C1")("--+-",   [kpts]*4, qtot=kpts[2], mod=mod)
    make("C2")("--0+0-", [kpts]*4, qtot=kpts[2], mod=mod, signs="--+-")

    kpts = fixt_symlabels_3D[(2,2,1)]

    make("D1")("+--",    [kpts]*3)
    make("D2")("+--",    [kpts]*3,               mod=mod)
    make("E1")("--+-",   [kpts]*4, qtot=kpts[2], mod=mod)
    make("E2")("--0+0-", [kpts]*4,               mod=mod, signs="--+-")

    return dct








class TestSymmetry3D:


   @pytest.fixture(autouse=True)
   def request_mod(self, fixt_mod_3D):
       self.mod = fixt_mod_3D



   @pytest.fixture(autouse=True)
   def request_symlabels(self, fixt_symlabels_3D):
       self.symlabels = fixt_symlabels_3D



   @pytest.fixture(autouse=True)
   def request_symmetries_and_data(self, fixt_symmetries_3D):
       self.symmetries_and_data = fixt_symmetries_3D



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

       out = sym.aux_symlabels(symindices[0], phase)     
       ans = lib.make_aux_symlabels(sym, symindices, phase)

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

   def test_sum_meshgrid(self, fixt_symlabels_3D):

       kpts      = fixt_symlabels_3D[(3,3,1)]
       symlabels = [kpts]*4

       out  = tnsym.symmetry.sum_meshgrid(*symlabels)
       ans  = lib.sum_meshgrid(*symlabels)

       util.assert_array_close(out, ans)



   def test_flatten(self, fixt_symlabels_3D):

       kpts      = fixt_symlabels_3D[(3,3,1)]
       symlabels = [kpts]*4

       summed_symlabels = lib.sum_meshgrid(*symlabels)

       out  = tnsym.symmetry.flatten(summed_symlabels)
       ans  = lib.flatten(summed_symlabels)

       util.assert_array_close(out, ans)



   def test_get_symlabels_shape(self):

       symlabels = np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                             [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                             [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])

       out = tnsym.symmetry.get_symlabels_shape([symlabels]*4)
       assert out == (9,9,9,9)



   def test_sum_abs_inner_symlabels(self): 

       symlabels = np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                             [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                             [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])

       ans = np.array([0,1,2,1,2,3,2,3,4])
       out = tnsym.symmetry.sum_abs_inner_symlabels(symlabels)
       util.assert_array_equal(out, ans)














































































































































































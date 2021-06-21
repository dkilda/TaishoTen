#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

import lib
import util

import taishoten as tn
from taishoten import Str



@pytest.fixture
def fixt_random_maps():

    dct = {}
    def make(key):
        def _make(*args, **kwargs):  
            dct[key] = lib.make_random_map(*args, **kwargs)
        return _make

    np.random.seed(1)

    make(0)(Str("IJK"),  (2,3,4),   5)
    make(1)(Str("MI"),   (3,2),     2)
    make(2)(Str("UJIW"), (5,3,2,5), 12)

    return dct




@pytest.fixture
def fixt_symmetries_1D():
    
    Si = np.arange(0,4)
    Sj = np.arange(1,3)
    Sk = np.arange(2,5)
    Sl = np.arange(0,2)
    Sm = np.arange(0,3)
    Sn = np.arange(1,5)

    dct = {}
    def make(key):
        def _make(*args, **kwargs):  
            dct[key] = lib.make_symmetry_1D(*args, **kwargs)
        return _make

    make("A1")("--+-",  [Si,Sj,Sk,Sl])
    make("B1")("++-",   [Sm,Sl,Sn])
    make("B2")("++-",   [Sm,Sl,Sn], qtot=2, mod=3)
    make("B3")("+0+0-", [Sm,Sl,Sn])
    
    return dct





    

class TestMap:

   @pytest.fixture(autouse=True)
   def request_random_maps_and_data(self, fixt_random_maps):
       self._random_maps_and_data = fixt_random_maps




   def random_maps_and_data(self, key):
       mp, data = self._random_maps_and_data[key]
       return mp, data




   def random_maps(self, key):
       mp, _ = self._random_maps_and_data[key]
       return mp




   @pytest.mark.parametrize("key", [0,1,2])
   def test_construct(self, key):

       mp, data = self.random_maps_and_data(key)
       
       legs  = data["legs"]
       shape = data["shape"]
       array = data["array"]

       util.assert_map_vs_array(mp, legs, array)
       util.assert_map(mp, legs, shape)




   @pytest.mark.parametrize("key, legs", [[0, Str("UML")], \
                                          [2, Str("WTYK")]])
   def test_set_legs(self, key, legs):

       mp = self.random_maps(key)
       mp._set_legs(legs)
       assert mp.legs == legs




   @pytest.mark.parametrize("key, legs", [[2, Str("KRUPT")], \
                                          [2, Str("LPR")]])
   @pytest.mark.xfail
   def test_set_legs_failed(self, key, legs):

       mp = self.random_maps(key)
       mp._set_legs(legs)




   @pytest.mark.parametrize("key", [0])
   def test_copy(self, key):

       mp  = self.random_maps(key)
       out = mp.copy()
       util.assert_map_equal(out, mp)




   @pytest.mark.parametrize("key", [0,1,2])
   def test_compute_from_idx(self, key):

       mp, data = self.random_maps_and_data(key)

       legs  = data["legs"]
       shape = data["shape"]
       array = data["array"]
       idx   = data["idx"]

       out = tn.Map.compute_from_idx(shape, idx, legs)

       util.assert_map_equal(out, mp)
       util.assert_map_vs_array(out, legs, array)
       util.assert_map(out, legs, shape)




   @pytest.mark.parametrize("key, legs", [["A1", Str("IJKL")], \
                                          ["B1", Str("MLN")],  \
                                          ["B2", Str("MLN")],  \
                                          ["B3", Str("MLN")]])
   def test_compute(self, fixt_symmetries_1D, key, legs):

       sym, _ = fixt_symmetries_1D[key]
       out    = tn.Map.compute(sym, legs)

       ans, data = lib.make_map_from_sym(legs, sym.signs, sym.symlabels, \
                                               qtot=sym.qtot, mod=sym.mod)

       util.assert_map_equal(out, ans)
       util.assert_map_vs_array(out, legs, data["array"])
       util.assert_map(out, legs, sym.shape)




   @pytest.mark.parametrize("keyA, keyB, legsC, shapeC",     \
                            [                                \
                             [0, 1, Str("JKM"),  (3,4,3)],   \
                             [0, 2, Str("IKUW"), (2,4,5,5)], \
                             [2, 0, Str("JUWK"), (3,5,5,4)], \
                            ])
   @pytest.mark.parametrize("which_method", [0,1])
   def test_contract_maps(self, keyA, keyB, legsC, shapeC, which_method):

       A = self.random_maps(keyA)
       B = self.random_maps(keyB)

       def contract(A, B, legsC):
           if   which_method == 0:
                return tn.contract_maps(A, B, legsC)
           else:
                return A.contract(B, legsC)

       subscript = tn.legs_to_subscript(A.legs, B.legs, legsC)
       ans       = lib.np_einsum_of_maps(subscript, legsC, A, B)

       out = contract(A, B, legsC)

       util.assert_map(out, legsC, shapeC)
       util.assert_map_vs_array(out, legsC, ans.array)
       util.assert_map_equal(out, ans)































































































































































































































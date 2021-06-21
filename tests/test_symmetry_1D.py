#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

import lib
import util
from lib import isiterable, noniterable

import taishoten as tn
import taishoten.symmetry as tnsym
from taishoten import Str




@pytest.fixture
def fixt_symlabels_1D():

    Si = np.arange(0,5)
    Sj = np.arange(0,2)
    Sk = np.arange(1,5)
    Sl = np.arange(0,4)

    return Si, Sj, Sk, Sl





@pytest.fixture
def fixt_symmetries_1D(fixt_symlabels_1D):
    
    Si, Sj, Sk, Sl = fixt_symlabels_1D

    dct = {}
    def make(key):
        def _make(*args, **kwargs):  
            dct[key] = lib.make_symmetry_1D(*args, **kwargs)
        return _make

    make("A1")("+--",  [Si,Sj,Sk])
    make("A2")("+0--", [Si,Sj,Sk], signs="+--")
    make("A3")("+--",  [Si,Sj,Sk], mod=2)
    make("A4")("+--",  [Si,Sj,Sk], mod=3)

    make("B1")("--+-",   [Sl,Sj,Si,Sk])
    make("B2")("--0+0-", [Sl,Sj,Si,Sk], signs="--+-")
    make("B3")("--00+-", [Sl,Sj,Si,Sk], signs="--+-")

    make("C1")("+-+-", [Sk,Sl,Sj,Si], 1, 2)
    make("C2")("+-+-", [Sk,Sl,Sj,Si], 1, None)
    
    return dct








class TestSymmetry1D:


   @pytest.fixture(autouse=True)
   def request_symlabels(self, fixt_symlabels_1D):
       self.symlabels = fixt_symlabels_1D



   @pytest.fixture(autouse=True)
   def request_symmetries_and_data(self, fixt_symmetries_1D):
       self.symmetries_and_data = fixt_symmetries_1D



   def symmetries(self, key):
       sym, _ = self.symmetries_and_data[key]
       return sym




   @pytest.mark.parametrize("key", ["A1", "A2", "A3", \
                                    "B1", "B2", "B3", \
                                    "C1", "C2"        \
                                   ])
   def test_construct(self, key):

       sym, data = self.symmetries_and_data[key] 
       util.assert_symmetry(sym, **data)
   



   @pytest.mark.parametrize("key", ["A1"])
   def test_construct_from_range_input(self, key):

       sym, data = self.symmetries_and_data[key]  
       qtot      = data["qtot"]
       mod       = data["mod"]   
       fullsigns = data["fullsigns"] 
       symlabels = [range(s[0], s[-1] + 1) for s in data["symlabels"]]

       sym = tn.Symmetry1D(fullsigns, symlabels, qtot, mod)
       util.assert_symmetry(sym, **data)




   @pytest.mark.parametrize("key, shape", [["A1", (5,2,4)  ], \
                                           ["A2", (5,2,4)  ], \
                                           ["A3", (5,2,4)  ], \
                                           ["B1", (4,2,5,4)], \
                                           ["B2", (4,2,5,4)], \
                                           ["B3", (4,2,5,4)], \
                                           ["C1", (4,4,2,5)], \
                                           ["C2", (4,4,2,5)], \
                                          ]) 
   def test_shape(self, key, shape):
       sym = self.symmetries(key) 
       assert sym.shape           == shape
       assert sym.truncated_shape == shape[:-1]




   @pytest.mark.parametrize("key, flipped_fullsigns, flipped_signs",    \
                             [                            \
                              ["A1",  "-++",    "-++"  ], \
                              ["A2",  "-0++",   "-++"  ], \
                              ["B1",  "++-+",   "++-+" ], \
                              ["B2",  "++0-0+", "++-+" ], \
                              ["B3",  "++00-+", "++-+" ], \
                              ["C1",  "-+-+",   "-+-+" ], \
                             ]) 
   def test_flipped_signs(self, key, flipped_fullsigns, flipped_signs):

       sym = self.symmetries(key) 
       assert sym.flipped_fullsigns == flipped_fullsigns
       assert sym.flipped_signs     == flipped_signs

   


   def test_map(self):

       sym = self.symmetries("A1")
       legs = Str("IJK")
       shape = (5,2,4)
       num_elems = 11

       np.random.seed(1)

       assert not sym.has_map
       mp, _ = lib.make_random_map(legs, shape, num_elems)
       sym.set_map(mp)

       assert sym.has_map
       util.assert_map_equal(sym.map, mp)
       util.assert_map(sym.map, legs, shape)
 
       sym.unset_map()
       assert not sym.has_map




   @pytest.mark.parametrize("shape, idx",                            \
                            [                                        \
                             [(10,),   [(1,3,7,8)                ]], \
                             [(10,12), [(0,2,3,6,7), (11,5,0,7,9)]], \
                            ])
   def test_find_zeros(self, shape, idx):

       sym = self.symmetries("A1")

       np.random.seed(1)

       A = lib.randn(*shape)
       for i in zip(*idx):
           A[i] = 0.0

       out  = sym.find_zeros(A)
       out1 = tnsym.symmetry.find_zeros(A)

       util.assert_array_equal(out,  idx[0])
       util.assert_array_equal(out1, idx[0])



   @pytest.mark.parametrize("shape", [10, (10,1), (10,3), \
                                       5,  (5,1),  (5,3), \
                                      13, (13,1), (13,3), \
                                     ])
   def test_sum_abs_inner_symlabels(self, shape):

       sym = self.symmetries("A1")

       np.random.seed(1)
       symlabels = lib.randint(-21, 21, shape)

       out = sym.sum_abs_inner_symlabels(symlabels)
       ans = abs(symlabels)
       if  symlabels.ndim > 1:
           ans = np.sum(abs(symlabels), axis=-1) 

       assert type(out) is np.ndarray
       assert all(noniterable(v) for v in out)
       assert all(v >= 0         for v in out)
       util.assert_array_equal(out, ans)




   @pytest.mark.parametrize("key, indices, phase", \
                            [
                             ["A1",  None,    None], \
                             ["B1",  None,    None], \
                             ["B1",  [1],     None], \
                             ["B1",  [0,2],   None], \
                             ["B1",  [0,2,3], None], \
                             ["B1",  [2,0,3], None], \
                             ["B1",  [2,0,3],   -1], \
                             ["B2",  None,    None], \
                             ["B2",  [1,0],   None], \
                             ["B2",  [1,2],     -1], \
                             ["B2",  [3,1,2],   -1], \
                            ]) 
   def test_flatten_symlabels(self, key, indices, phase):

       sym = self.symmetries(key)

       if  indices is None:
           indices = range(sym.num_symlegs)

       if  phase is None:
           phase = 1

       signs     = ''.join([sym.signs[i]     for i in indices]) 
       symlabels =         [sym.symlabels[i] for i in indices] 

       out  = sym.flatten_symlabels(indices, phase)
       ans  = lib.flatten_symlabels(signs,   symlabels,   phase)
       ans1 = lib.flatten_symlabels_1(signs, symlabels, phase)
       ans2 = lib.flatten_symlabels_2(signs, symlabels, phase)

       assert type(out) is np.ndarray
       assert out.shape == (lib.sym_size(symlabels), 1)

       util.assert_array_close(out, ans)
       util.assert_array_close(out, ans1)
       util.assert_array_close(out, ans2)




   @pytest.mark.parametrize("key, symindices, phase", \
                            [
                             ["A1",  ([0,2],   [1]),    1], \
                             ["A1",  ([0,2],   [1]),   -1], \
                             ["B1",  ([0,3],   [1,2]),  1], \
                             ["B2",  ([0,3],   [1,2]),  1], \
                             ["B2",  ([0,3,2], [1]),    1], \
                             ["C1",  ([0,3],   [1,2]),  1], \
                            ]) 
   def test_aux_symlabels(self, key, symindices, phase):

       sym = self.symmetries(key) 

       out = sym.aux_symlabels(symindices[0], phase)     
       ans = lib.make_aux_symlabels(sym, symindices, phase)

       assert type(out) is np.ndarray
       util.assert_array_close(out, ans)




   @pytest.mark.parametrize("key, indices, which", [["A1", [0,2], 1]])
   def test_aux_symlabels_by_hand(self, key, indices, which):

       sym = self.symmetries(key)

       ans = self.symlabels[which]
       out = sym.aux_symlabels(indices)

       assert type(out) is np.ndarray
       util.assert_array_close(out, ans)



   @pytest.mark.parametrize("key, symlabels, aligned_symlabels",           \
           [                                                               \
            ["A1", [np.arange(0,5), np.arange(0,2)], np.array([0,1])],     \
            ["A1", [np.arange(0,5), np.arange(1,5)], np.array([1,2,3,4])], \
           ])
   def test_align_symlabels(self, key, symlabels, aligned_symlabels):
       
       sym = self.symmetries(key)

       out = sym.align_symlabels(*symlabels)
       ans = lib.align_symlabels(*symlabels)

       util.assert_array_close(out, ans)
       util.assert_array_close(out, aligned_symlabels)

       out1 = sym.align_symlabels(*symlabels[::-1])
       ans1 = lib.align_symlabels(*symlabels[::-1])

       util.assert_array_close(out1, ans1)


        
   @pytest.mark.parametrize("key, signsA, A, B", \
           [["B1", '-+', [np.arange(0,4), np.arange(0,5)], np.arange(1,5)]])
   def test_align_symlabels_1(self, key, signsA, A, B):
       
       sym = self.symmetries(key)

       flatA = lib.flatten_symlabels(signsA, A)
       B     = B if B.ndim > 1 else B.reshape((len(B), 1))

       out = sym.align_symlabels(flatA, B)
       ans = lib.align_symlabels(flatA, B)

       out1 = sym.align_symlabels(B, flatA)
       ans1 = lib.align_symlabels(B, flatA)

       util.assert_array_close(out,  ans)
       util.assert_array_close(out1, ans1)
  


   @pytest.mark.parametrize("key, symlabels, ans",                 \
           [                                                       \
            ["A4", np.arange(0,5),         np.array([0,1,2,0,1])], \
            ["A3", np.array([-1,2,-3,-4]), np.array([1,0,1,0])],   \
           ])
   def test_apply_mod(self, key, symlabels, ans):

       sym = self.symmetries(key)
       out = sym.apply_mod(symlabels)
       util.assert_array_equal(out, ans)



   @pytest.mark.parametrize("key, symlabels, ans",             \
           [                                                   \
            ["A4", np.arange(0,5),         np.array([0,1,2])], \
            ["A3", np.array([-1,2,-3,-4]), np.array([0,1])],   \
           ])
   def test_fold(self, key, symlabels, ans):

       sym = self.symmetries(key)
       out = sym.fold(symlabels)
       util.assert_array_equal(out, ans)

      



class TestSymmetryUtil:

   def test_sum_meshgrid(self):

       symlabels = [np.arange(0,5),  np.arange(0,2), \
                   -np.arange(1,5), -np.arange(0,4)]

       out  = tnsym.symmetry.sum_meshgrid(*symlabels)
       ans  = lib.sum_meshgrid(*symlabels)
       ans1 = lib.sum_meshgrid_1(*symlabels)

       util.assert_array_close(out, ans)
       util.assert_array_close(out, ans1)



   def test_flatten(self):

       symlabels = [np.arange(0,5),  np.arange(0,2), \
                   -np.arange(1,5), -np.arange(0,4)]

       summed_symlabels = lib.sum_meshgrid(*symlabels)

       out  = tnsym.symmetry.flatten(summed_symlabels)
       ans  = lib.flatten(summed_symlabels)
       ans1 = lib.flatten_1(summed_symlabels)

       util.assert_array_close(out, ans)
       util.assert_array_close(out, ans1)



   def test_set_symtol(self):

       out = tnsym.symmetry.set_symtol()
       util.assert_array_close(out, 2**(-16))

       out = tnsym.symmetry.set_symtol(1e-10)
       util.assert_array_close(out, 1e-10)



   def test_get_symlabels_shape(self):

       symlabels = [np.arange(0,5), -np.arange(0,2), \
                    np.arange(1,5), -np.arange(2,9), np.arange(0,3)]

       out = tnsym.symmetry.get_symlabels_shape(symlabels)
       assert out == (5,2,4,7,3)



   @pytest.mark.parametrize("symlabels, ans",                                \
   [                                                                         \
   [np.array([[1,0,-2], [3,5,0], [2,1,1], [3,-4,3]]), np.array([3,8,4,10])], \
   [np.array([[1], [3], [-2], [3]]),                  np.array([1,3,2,3]) ], \
   [np.array([1, 3, -2, 3]),                          np.array([1,3,2,3]) ], \
   ])
   def test_sum_abs_inner_symlabels(self, symlabels, ans): 

       out = tnsym.symmetry.sum_abs_inner_symlabels(symlabels)
       util.assert_array_equal(out, ans)













































































































































#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import helper_lib as lib
import taishoten as tn

from taishoten import Str
from .util     import TaishoTenTestCase, must_fail



class TestSymmetry1D(TaishoTenTestCase):


   def test_construct(self):

       def _test(fullsigns, symlabels, qtot=0, mod=None, signs=None):

           sym = tn.Symmetry1D(fullsigns, symlabels, qtot, mod)
           self.assertSymmetry(sym, fullsigns, symlabels, qtot, mod, signs)

       Si = range(0,5)
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)

       _test("+--",  [Si,Sj,Sk])
       _test("--+-", [Sl,Sj,Si,Sk])
       _test("+-+-", [Sk,Sl,Sj,Si], 1, None)
       _test("+-+-", [Sk,Sl,Sj,Si], 1, 2)

       _test("+0--",   [Si,Sj,Sk],    signs="+--")
       _test("--0+0-", [Sl,Sj,Si,Sk], signs="--+-")
       _test("--00+-", [Sl,Sj,Si,Sk], signs="--+-")



   def test_shape(self):

       def _test(shape, fullsigns, symlabels, qtot=0, mod=None):

           sym  = tn.Symmetry1D(fullsigns, symlabels, qtot, mod)
           out  = sym.shape
           out1 = sym.truncated_shape

           self.assertEqual(out,  shape)
           self.assertEqual(out1, shape[:-1])

       Si = range(0,5)
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)

       _test((5,2,4),   "+--",  [Si,Sj,Sk])
       _test((4,2,5,4), "--+-", [Sl,Sj,Si,Sk])
       _test((4,4,2,5), "+-+-", [Sk,Sl,Sj,Si], 1, 2)

       _test((5,2,4),   "+0--",   [Si,Sj,Sk])
       _test((4,2,5,4), "--0+0-", [Sl,Sj,Si,Sk])
       _test((4,2,5,4), "--00+-", [Sl,Sj,Si,Sk])



   def test_flipped_signs(self):

       def _test(sym, flipped_fullsigns, flipped_signs=None):

           if flipped_signs is None:
              flipped_signs = flipped_fullsigns

           out  = sym.flipped_fullsigns
           out1 = sym.flipped_signs

           assert out  == flipped_fullsigns
           assert out1 == flipped_signs

       Si = range(0,5)
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)

       symA = tn.Symmetry1D("+--",    [Si,Sj,Sk])
       symB = tn.Symmetry1D("--+-",   [Sl,Sj,Si,Sk])
       symC = tn.Symmetry1D("+-+-",   [Sk,Sl,Sj,Si], 1, 2)
       symD = tn.Symmetry1D("+0--",   [Si,Sj,Sk])
       symE = tn.Symmetry1D("--0+0-", [Sl,Sj,Si,Sk])
       symF = tn.Symmetry1D("--00+-", [Sl,Sj,Si,Sk])

       _test(symA, "-++")
       _test(symB, "++-+")
       _test(symC, "-+-+")
       _test(symD, "-0++",   flipped_signs="-++")
       _test(symE, "++0-0+", flipped_signs="++-+")
       _test(symF, "++00-+", flipped_signs="++-+")

       

   def test_map(self):

       def _test(sym, legs, shape, num_elems):

           assert not sym.has_map
           mp = lib.create_random_map(legs, shape, num_elems)
           sym.set_map(mp)

           assert sym.has_map
           self.assertEqualMap(sym.map, mp)
           self.assertMap(sym.map, legs, shape)
 
           sym.unset_map()
           assert not sym.has_map

       # Define symmetry labels
       np.random.seed(1)

       Si = range(0,5)
       Sj = range(0,2)
       Sk = range(1,5)

       # Test
       sym = tn.Symmetry1D("+--",  [Si,Sj,Sk])
       _test(sym, "IJK", (5,2,4), 5)       



   def test_find_zeros(self):

       def _test(sym, shape, idx):

           A = lib.randn(shape)
           for i in idx:
               A[i] = 0.0

           out = sym.find_zeros(A)
           self.assertEqual(out, idx[0])

       # Define symmetry
       np.random.seed(1)

       Si = range(0,5)
       Sj = range(0,2)
       Sk = range(1,5)
       sym = tn.Symmetry1D("+--",  [Si,Sj,Sk])

       # Test-1
       idx   = [(1,3,7,8)]
       shape = (10,)
       _test(sym, shape, idx)

       # Test-2
       idx   = [(0,2,3,6,7), (11,19,0,7,9)]
       shape = (10,12)
       _test(sym, shape, idx)
       


   def test_sum_abs_inner_symlabels(self):

       def _test(sym):

          out = sym.sum_abs_inner_symlabels(sym.symlabels)
          ans = [np.sum(s, axis=-1) for s in sym.symlabels]
          self.assertEqualArrayList(out, ans)

       np.random.seed(1)

       # Test-1
       symlabels = [lib.randn(5), lib.randn(2), lib.randn(4)]
       sym       = tn.Symmetry1D("+--", symlabels)
       _test(sym)

       # Test-2
       symlabels = [lib.randn(5,3), \
                    lib.randn(2,3), \
                    lib.randn(4,3)  ]

       sym = tn.Symmetry3D("+--", symlabels)
       _test(sym)



   def test_flatten_symlabels(self):

       def _test(sym, indices=None, phase=1):
            
           if  indices is None:
               indices = range(len(sym.shape))
   
           signs     = ''.join([sym.signs[i]     for i in indices]) 
           symlabels =         [sym.symlabels[i] for i in indices] 

           out  = sym.flatten_symlabels(indices, phase)
           ans  = lib.flatten_symlabels(signs, symlabels, phase)
           ans1 = lib.flatten_symlabels_1(signs, symlabels, phase)
           ans2 = lib.flatten_symlabels_2(signs, symlabels, phase)

           assert type(out) is np.ndarray

           size = np.prod([len(s) for s in symlabels])
           self.assertEqual(out.shape, (size, 1))

           self.assertCloseArray(out, ans)
           self.assertCloseArray(out, ans1)
           self.assertCloseArray(out, ans2)

       # Define symmetries
       Si = np.arange(0,5)
       Sj = np.arange(0,2)
       Sk = np.arange(1,5)
       Sl = np.arange(0,4)
         
       symA = tn.Symmetry1D("+--",    [Si,Sj,Sk])
       symB = tn.Symmetry1D("--+-",   [Sl,Sj,Si,Sk])
       symC = tn.Symmetry1D("--0+0-", [Sl,Sj,Si,Sk])

       # Main tests
       _test(symA)
       _test(symB)
       _test(symC)

       _test(symB, indices=[1])
       _test(symB, indices=[0,2])
       _test(symB, indices=[0,2,3])
       _test(symB, indices=[2,0,3], phase=-1)

       _test(symC, indices=[0,1])
       _test(symC, indices=[1,3],   phase=-1)
       _test(symC, indices=[1,3,5])

       # Corner cases
       must_fail(_test, ValueError)(symB, indices=[])

       out = symB.flatten_symlabels([1])
       assert type(out) is np.ndarray
       self.assertEqualArray(out, np.array(Sj))



   def test_aux_symlabels(self):

       def _test(sym, signs, symlabels, indices, phase=1):

           out = sym.aux_symlabels(indices, phase)     
           ans = lib.make_aux_symlabels(signs, symlabels, \ 
                                        sym.qtot, sym.mod, phase)

           assert type(out) is np.ndarray

           size = np.prod([len(s) for s in symlabels[0]])
           self.assertEqual(out.shape, (size, 1))
           self.assertCloseArray(out, ans)

       # Define symlabels
       Si = np.arange(0,5)
       Sj = np.arange(0,2)
       Sk = np.arange(1,5)
       Sl = np.arange(0,4)

       # Test-0
       sym     = tn.Symmetry1D("+--", [Si,Sj,Sk])
       indices = [0,2]
       out = sym.aux_symlabels(indices)
       ans = np.array(Sj)
       self.assertCloseArray(out, ans)

       # Test-1
       sym       = tn.Symmetry1D("+--", [Si,Sj,Sk])
       signs     = (["+-"],  ["-"])
       symlabels = ([Si,Sk], [Sj])
       indices   = [0,2]
       _test(sym, signs, symlabels, indices)

       # Test-2
       sym       = tn.Symmetry1D("--+-",   [Sl,Sj,Si,Sk])
       signs     = (["--"],  ["-+"])
       symlabels = ([Sl,Sk], [Sj,Si])
       indices   = [0,3]
       _test(sym, signs, symlabels, indices)

       # Test-3
       sym       = tn.Symmetry1D("--+-",   [Sl,Sj,Si,Sk], qtot=1, mod=2)
       signs     = (["--"],  ["-+"])
       symlabels = ([Sl,Sk], [Sj,Si])
       indices   = [0,3]
       _test(sym, signs, symlabels, indices, phase=-1)

       # Test-4
       sym       = tn.Symmetry1D("--0+0-", [Sl,Sj,Si,Sk])
       signs     = (["--"], ["-+"])
       symlabels = ([Sl,Sk], [Sj,Si])
       indices   = [0,5]
       _test(sym, signs, symlabels, indices)

       # Test-5
       sym       = tn.Symmetry1D("--0+0-", [Sl,Sj,Si,Sk])
       signs     = (["--+"], ["-"])
       symlabels = ([Sl,Sk,Si], [Sj])
       indices   = [0,5,3]
       _test(sym, signs, symlabels, indices)



   def test_align_symlabels(self):

       # Define symmetries
       Si = np.arange(0,5)
       Sj = np.arange(0,2)
       Sk = np.arange(1,5)
       Sl = np.arange(0,4)

       symA = tn.Symmetry1D("+--",  [Si,Sj,Sk])
       symB = tn.Symmetry1D("--+-", [Sl,Sj,Si,Sk])

       # Test-1
       out  = symA.align_symlabels(Si, Sj)
       ans  = lib.align_symlabels(Si, Sj)
       ans1 = np.array([0,1])
       self.assertCloseArray(out, ans)
       self.assertCloseArray(out, ans1)

       # Test-2
       out  = symA.align_symlabels(Si, Sk)
       ans  = lib.align_symlabels(Si, Sk)
       ans1 = np.array([1,2,3,4])
       self.assertCloseArray(out, ans)
       self.assertCloseArray(out, ans1)

       # Test-3 
       Sli = lib.flatten_symlabels("-+", [Sl,Si])
       out = symB.align_symlabels(Sli, Sk)
       ans = lib.align_symlabels(Sli, Sk)
       self.assertCloseArray(out, ans)

       # Test-4
       out = symB.align_symlabels(Sk, Sli)
       ans = lib.align_symlabels(Sk, Sli)
       self.assertCloseArray(out, ans)      
      
             

   def test_apply_mod(self):

       # Define symmetries
       Si = np.arange(0,5)
       Sj = np.arange(0,2)
       Sk = np.arange(1,5)

       symA = tn.Symmetry1D("+--", [Si,Sj,Sk], mod=3)
       symB = tn.Symmetry1D("+--", [Si,Sj,Sk], mod=2)

       # Test-1
       out = symA.apply_mod(np.arange(0,5))
       ans = np.array([0,1,2,0,1])
       self.assertEqualArray(out, ans)

       # Test-2
       out = symA.apply_mod(np.array([-1,2,-3,-4]))
       ans = np.array([1,0,1,0])
       self.assertEqualArray(out, ans)



   def test_fold(self):

       # Define symmetries
       Si = np.arange(0,5)
       Sj = np.arange(0,2)
       Sk = np.arange(1,5)

       symA = tn.Symmetry1D("+--", [Si,Sj,Sk], mod=3)
       symB = tn.Symmetry1D("+--", [Si,Sj,Sk], mod=2)

       # Test-1
       out = symA.fold(np.arange(0,5))
       ans = np.array([0,1,2])
       self.assertEqualArray(out, ans)

       # Test-2
       out = symA.fold(np.array([-1,2,-3,-4]))
       ans = np.array([0,1])
       self.assertEqualArray(out, ans)






class TestSymmetryUtil(TaishoTenTestCase):


   def test_sum_meshgrid(self):

       Si =  np.arange(0,5)
       Sj =  np.arange(0,2)
       Sk = -np.arange(1,5)
       Sl = -np.arange(0,4)

       out  = tn.symmetry.sum_meshgrid(Si,Sj,Sk,Sl)
       ans  = lib.sum_meshgrid(Si,Sj,Sk,Sl)
       ans1 = lib.sum_meshgrid_1(Si,Sj,Sk,Sl)

       self.assertEqualArray(out, ans)
       self.assertEqualArray(out, ans1)



   def test_flatten(self):

       Si =  np.arange(0,5)
       Sj =  np.arange(0,2)
       Sk = -np.arange(1,5)
       Sl = -np.arange(0,4)

       Sijkl = lib.sum_meshgrid(Si,Sj,Sk,Sl)

       out   = tn.symmetry.flatten(Sijkl)
       ans   = lib.flatten(Sijkl)
       ans_1 = lib.flatten_1(Sijkl)

       self.assertEqualArray(out, ans)
       self.assertEqualArray(out, ans1)



   def test_find_zeros(self):

       def _test(shape, idx):

           A = lib.randn(shape)
           for i in idx:
               A[i] = 0.0

           out = tn.symmetry.find_zeros(A)
           self.assertEqual(out, idx[0])

       np.random.seed(1)

       # Test-1
       idx   = [(1,3,7,8)]
       shape = (10,)
       _test(shape, idx)

       # Test-2
       idx   = [(0,2,3,6,7), (11,19,0,7,9)]
       shape = (10,12)
       _test(sshape, idx)



   def test_set_symtol(self):

       out = tn.symmetry.set_symtol()
       self.assertClose(out, 2**(-16))

       out = tn.symmetry.set_symtol(1e-10)
       self.assertClose(out, 1e-10)



   def test_get_symlabels_shape(self):

       Si = range(0,5)
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(2,9)
       Sm = range(0,3)

       ans = (5,2,4,7,3)
       out = tn.symmetry.get_symlabels_shape([Si,Sj,Sk,Sl,Sm])
       assert out == ans



   def test_sum_abs_inner_symlabels(self): 

       Si = np.array([[1,0,-2], [3,5,0], [2,1,1], [3,-4,3]])
       Sj = np.array([[1], [3], [-2], [3]])
       Sk = np.array([1,3,-2,3])

       # Test-1
       ans = np.array([3,8,4,10])
       out = tn.symmetry.sum_abs_inner_symlabels(Si)
       self.assertEqualArray(out, ans)

       # Test-2
       ans = np.array([1,3,2,3])
       out = tn.symmetry.sum_abs_inner_symlabels(Sj)
       self.assertEqualArray(out, ans)

       # Test-3
       ans = np.array([1,3,2,3])
       out = tn.symmetry.sum_abs_inner_symlabels(Sk)
       self.assertEqualArray(out, ans)























































































































#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import helper_lib as lib
import taishoten as tn

from taishoten import Str
from .util     import TaishoTenTestCase
from .util     import must_fail, CLOSE_RTOL



class TestSymmetry3D(TaishoTenTestCase):

   def test_construct(self):

       def _test(fullsigns, symlabels, qtot=0, mod=None, signs=None):

           sym = tn.Symmetry3D(fullsigns, symlabels, qtot, mod)
           self.assertSymmetry(sym, fullsigns, symlabels, qtot, mod, signs)

       # Tests
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [3,3,1])

       _test("+--",    [kpts]*3)
       _test("+--",    [kpts]*3,               mod=mod)
       _test("+---",   [kpts]*4, qtot=kpts[2], mod=mod)
       _test("+0--0-", [kpts]*4, qtot=kpts[2], mod=mod, signs="+---")

       # Tests that must fail
       wrong_mod  = np.ones(2,3)
       wrong_kpts = lib.make_symlabels_3D(mod, [3,3])

       must_fail(_test)("+--", [kpts]*3,       mod=wrong_mod)    
       must_fail(_test)("+--", [wrong_kpts]*3, mod=mod)  
       must_fail(_test)("+--", [wrong_kpts]*3, mod=wrong_mod)  

   

   def test_shape(self):

       def _test(shape, fullsigns, symlabels, qtot=0, mod=None):

           sym  = tn.Symmetry3D(fullsigns, symlabels, qtot, mod)
           out  = sym.shape
           out1 = sym.truncated_shape

           self.assertEqual(out,  shape)
           self.assertEqual(out1, shape[:-1])

       # Test-1
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [2,2,1])

       _test((4,4,4),   "+--",    [kpts]*3)
       _test((4,4,4,4), "+---",   [kpts]*4, qtot=kpts[2], mod=mod)
       _test((4,4,4,4), "+0--0-", [kpts]*4,               mod=mod)  
       
       # Test-2
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [3,3,1])

       _test((9,9,9),   "+--",    [kpts]*3)
       _test((9,9,9,9), "+---",   [kpts]*4, qtot=kpts[2], mod=mod)
       _test((9,9,9,9), "+0--0-", [kpts]*4,               mod=mod)       



   def test_sum_abs_inner_symlabels(self):

       def _test(sym):

          out = sym.sum_abs_inner_symlabels(sym.symlabels)
          ans = [np.sum(s, axis=-1) for s in sym.symlabels]
          self.assertCloseArrayList(out, ans)

       # Test-1
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [2,2,1])
       sym  = tn.Symmetry3D("+--", kpts, mod=mod)

       _test(sym)

       # Test-2
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [3,3,1])
       sym  = tn.Symmetry3D("+--", kpts, mod=mod)

       _test(sym)



   def test_flatten_symlabels(self):

       def _test(sym, indices=None, phase=1):
            
           if  indices is None:
               indices = range(len(sym.shape))
   
           signs     = ''.join([sym.signs[i]     for i in indices]) 
           symlabels =         [sym.symlabels[i] for i in indices] 

           out  = sym.flatten_symlabels(indices, phase)
           ans  = lib.flatten_symlabels(signs, symlabels, phase)

           assert type(out) is np.ndarray

           size = np.prod([len(s) for s in symlabels])
           self.assertEqual(out.shape, (size, 3))
           self.assertCloseArray(out, ans)

       # Define symmetries
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [3,3,1])

       symA = tn.Symmetry3D("+--",    [kpts]*3, mod=mod)
       symB = tn.Symmetry3D("--+-",   [kpts]*3, mod=mod)
       symC = tn.Symmetry3D("--0+0-", [kpts]*4, mod=mod)

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
       self.assertEqualArray(out, kpts)    



   def test_aux_symlabels(self):

       def _test(sym, signs, symlabels, indices, phase=1):

           out = sym.aux_symlabels(indices, phase)     
           ans = lib.make_aux_symlabels(signs, symlabels, \ 
                                        sym.qtot, sym.mod, phase)

           assert type(out) is np.ndarray

           size = np.prod([len(s) for s in symlabels[0]])
           self.assertEqual(out.shape, (size, 3))
           self.assertCloseArray(out, ans)

       # Define symlabels
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [3,3,1])

       # Test-0
       sym     = tn.Symmetry3D("+--", [kpts]*3)
       indices = [0,2]
       out = sym.aux_symlabels(indices)
       ans = kpts
       self.assertCloseArray(out, ans)

       # Test-1
       sym       = tn.Symmetry3D("+--", [kpts]*3, mod=mod)
       signs     = (["+-"],  ["-"])
       symlabels = ([kpts, kpts], [kpts])
       indices   = [0,2]
       _test(sym, signs, symlabels, indices)

       # Test-1.1
       sym       = tn.Symmetry3D("+--", [kpts]*3)
       signs     = (["+-"],  ["-"])
       symlabels = ([kpts, kpts], [kpts])
       indices   = [0,2]
       _test(sym, signs, symlabels, indices)

       # Test-2
       sym       = tn.Symmetry3D("--+-", [kpts]*4, mod=mod)
       signs     = (["--"],  ["-+"])
       symlabels = ([kpts, kpts], [kpts, kpts])
       indices   = [0,3]
       _test(sym, signs, symlabels, indices)

       # Test-3
       sym       = tn.Symmetry3D("--+-", [kpts]*4, qtot=kpts[2], mod=mod)
       signs     = (["--"],  ["-+"])
       symlabels = ([kpts, kpts], [kpts, kpts])
       indices   = [0,3]
       _test(sym, signs, symlabels, indices, phase=-1)

       # Test-4
       sym       = tn.Symmetry3D("--0+0-", [kpts]*4, mod=mod)
       signs     = (["--"], ["-+"])
       symlabels = ([kpts, kpts], [kpts, kpts])
       indices   = [0,5]
       _test(sym, signs, symlabels, indices)

       # Test-5
       sym       = tn.Symmetry3D("--0+0-", [kpts]*4, mod=mod)
       signs     = (["--+"], ["-"])
       symlabels = ([kpts, kpts, kpts], [kpts, kpts])
       indices   = [0,5,3]
       _test(sym, signs, symlabels, indices)
   


   def test_apply_mod(self):
    
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [2,2,1])
       symA = tn.Symmetry3D("+--", [kpts]*3, mod=mod)

       # Test-1
       array = np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                         [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                         [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])

       scaled_symlabels = (1/3) * array
       symlabels        = (2 * np.pi / 5) * scaled_symlabels

       out = symA.apply_mod(symlabels)
       ans = np.array([[ 0, 0, 0], [ 0, 1, 0], [ 0, 1, 0], \
                       [-1, 0, 0], [ 1,-1, 0], [ 1,-1, 0], \
                       [-1, 0, 0], [ 1, 1, 0], [-1, 1, 0]])
       ans = (1/3) * ans

       self.assertEqualArray(out, ans)
       
       

   def test_fold(self):

       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [2,2,1])
       symA = tn.Symmetry3D("+--", [kpts]*3, mod=mod)

       # Test-1
       array = np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                         [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                         [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])

       scaled_symlabels = (1/3) * array
       symlabels        = (2*np.pi/5) * scaled_symlabels

       out = symA.fold(symlabels)
       ans = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], 
                       [1, 2, 0], [2, 0, 0], [2, 1, 0]])
       ans = (1/3) * ans
       ans = np.round_(ans, 10)

       self.assertEqualArray(out, ans)







class TestSymmetryUtil(TaishoTenTestCase):


   def test_sum_meshgrid(self):

       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [3,3,1])

       out = tn.symmetry.sum_meshgrid(*[kpts]*4)
       ans = lib.sum_meshgrid(*[kpts]*4)
       self.assertCloseArray(out, ans)



   def test_flatten(self):

       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [3,3,1])

       summed_kpts = lib.sum_meshgrid(*[kpts]*4)

       out   = tn.symmetry.flatten(summed_kpts)
       ans   = lib.flatten(summed_kpts)
       self.assertCloseArray(out, ans)



   def test_invert_matrix_3D(self):

       # Test-1
       x   = (2 * np.pi / 5)   * np.eye(3)
       ans = (5 / (2 * np.pi)) * np.eye(3)
       out = tn.symmetry.invert_matrix_3D(x)
       self.assertCloseArray(out, ans)

       # Test-2
       x = (1 / np.sqrt(6)) * np.array([[ 1, 1,-2], \
                                        [-1, 2, 1], \
                                        [ 2,-1, 1]] )
       ans = np.linalg.inv(x)
       out = tn.symmetry.invert_matrix_3D(x)
       self.assertCloseArray(out, ans)



   def test_get_symlabels_shape(self):

       symlabels = np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                             [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                             [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])
       ans = (9,9,9,9)
       out = tn.symmetry.test_get_symlabels_shape([symlabels]*4)
       assert out == ans



   def test_sum_abs_inner_symlabels(self): 

       symlabels = np.array([[ 0, 0, 0], [ 0, 1, 0], [0,-2, 0], \
                             [-1, 0, 0], [ 1,-1, 0], [1, 2, 0], \
                             [ 2, 0, 0], [-2, 1, 0], [2,-2, 0]])

       ans = np.array([0,1,2,1,2,3,2,3,4])
       out = tn.symmetry.sum_abs_inner_symlabels(symlabels)
       self.assertEqualArray(out, ans)





class TestSymmetryContraction3D:

   # --- Test constructor --------------------------------------------------- #

   def test_construct(self):

       def _test(symA, symB, legs, symlegs, phase):

           sym = tn.dictriplet(symA, symB, None)
           out = tn.SymmetryContraction(symA, symB, legs)
           self.assertSymmetryContraction(out, sym, legs, symlegs, phase)   

       # Define symlabels
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [3,3,1])

       # Test-1
       symA    = tn.Symmetry3D("++-", [kpts]*3, mod=mod)
       symB    = tn.Symmetry3D("++-", [kpts]*3, mod=mod)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), None)
       phase   = 1

       _test(symA, symB, legs, symlegs, phase)

       # Test-2
       symA    = tn.Symmetry3D("+++-", [kpts]*4, mod=mod)
       symB    = tn.Symmetry3D("++-",  [kpts]*3, mod=mod)
       legs    = tn.dictriplet(Str("inom"), Str("inp"), Str("pom"))
       symlegs = tn.dictriplet(Str("INOM"), Str("INP"), None)
       phase = -1

       _test(symA, symB, legs, symlegs, phase)

       # Test-3
       symA    = tn.Symmetry3D("+0+0+-", [kpts]*4, mod=mod)
       symB    = tn.Symmetry3D("++00-",  [kpts]*3, mod=mod)
       legs    = tn.dictriplet(Str("ixnyom"), Str("inzxp"), Str("pyzom"))
       symlegs = tn.dictriplet(Str("INOM"),   Str("INP"),   None)
       phase = -1

       _test(symA, symB, legs, symlegs, phase)        

       # Test-4      
       Si = np.arange(2)
       symA    = tn.Symmetry3D("++-", [kpts]*3, mod=mod)
       symB    = tn.Symmetry1D("++-", [Si]*3)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), None)
       phase   = 1
     
       must_fail(_test)(symA, symB, legs, symlegs, phase)

       # Test-5
       symA    = tn.Symmetry3D("++-", [kpts]*3, mod=mod)
       symB    = tn.Symmetry3D("++-", [kpts]*3, mod=2*mod)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), None)
       phase   = 1

       must_fail(_test)(symA, symB, legs, symlegs, phase)



   # --- Test compute() ----------------------------------------------------- #

   def test_construct(self):

       def _test(symA, symB, symC, legs, symlegs, phase):

           sym = tn.dictriplet(symA, symB, symC)
           out = tn.SymmetryContraction(symA, symB, legs)
           out.compute()
           self.assertSymmetryContraction(out, sym, legs, symlegs, phase)   

       # Define symlabels
       mod  = (2 * np.pi / 5) * np.eye(3)
       kpts = lib.make_symlabels_3D(mod, [3,3,1])

       # Test-1
       symA    = tn.Symmetry3D("++-",  [kpts]*3,  mod=mod)
       symB    = tn.Symmetry3D("++-",  [kpts]*3,  mod=mod)
       symC    = tn.Symmetry3D("+++-", [kpts]*4,  mod=mod)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), Str("IJLM"))
       phase   = 1

       _test(symA, symB, symC, legs, symlegs, phase)

       # Test-2
       symA    = tn.Symmetry3D("+++-", [kpts]*4,  mod=mod)
       symB    = tn.Symmetry3D("++-",  [kpts]*3,  mod=mod)
       symC    = tn.Symmetry3D("--+",  [kpts]*3,  mod=mod)
       legs    = tn.dictriplet(Str("inom"), Str("inp"), Str("pom"))
       symlegs = tn.dictriplet(Str("INOM"), Str("INP"), Str("POM"))
       phase = -1

       _test(symA, symB, symC, legs, symlegs, phase)

       # Test-3
       symA    = tn.Symmetry3D("+0+0+-", [kpts]*4,  mod=mod)
       symB    = tn.Symmetry3D("++00-",  [kpts]*3,  mod=mod)
       symC    = tn.Symmetry3D("-00-+",  [kpts]*3,  mod=mod)
       legs    = tn.dictriplet(Str("ixnyom"), Str("inzxp"), Str("pyzom"))
       symlegs = tn.dictriplet(Str("INOM"),   Str("INP"),   Str("POM"))
       phase = -1

       _test(symA, symB, symC, legs, symlegs, phase)        


































































































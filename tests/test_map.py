#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import helper_lib as lib
import taishoten as tn

from taishoten import Str
from .util import TaishoTenTestCase, must_fail



class TestMap(TaishoTenTestCase):


   def test_construct(self):

       def _test(legs, shape, num_elems):

           array = lib.random_map_from_idx(shape, num_elems)[0]
           out = tn.Map(array, legs)  
     
           self.assertMapVsArray(out, Str(legs), array)
           self.assertMap(out, Str(legs), shape)

       np.random.seed(1)

       _test(Str("IJK"),  (2,3,4),   5)
       _test("MI",        (3,2),     2)
       _test(Str("UJIW"), (5,3,2,5), 10)



   def test_set_legs(self):

       np.random.seed(1)

       mapA = lib.create_random_map(Str("IJK"),  (2,3,4),   5)
       mapC = lib.create_random_map(Str("UJIW"), (5,3,2,5), 10)

       mapA.set_legs("UML")
       assert mapA.legs == Str("UML")

       mapC.set_legs(Str("WTYK"))
       assert mapC.legs == Str("WTYK")

       must_fail(mapC.set_legs)(Str("KRUPT"))
       must_fail(mapC.set_legs)(Str("LPR"))



 

   def test_copy(self):

       np.random.seed(1)

       ans = lib.create_random_map(Str("IJK"), (2,3,4), 5)[0]
       out = ans.copy()
       self.assertEqualMap(out, ans)



   def test_compute_from_idx(self):

       def _test(legs, shape, num_elems):

           array, idx = lib.random_map_from_idx(shape, num_elems)
           ans = tn.Map(array, legs)   
           out = tn.Map.compute_from_idx(shape, idx, legs)

           self.assertEqualMap(out, ans)
           self.assertMapVsArray(out, legs, array)
           self.assertMap(out, legs, shape)

       np.random.seed(1)
  
       _test(Str("IJK"),  (2,3,4),   5)
       _test("MI",        (3,2),     2)
       _test(Str("UJIW"), (5,3,2,5), 10)


 
   def test_compute(self):

       def _test(legs, fullsigns, symlabels, qtot=0, mod=None):

           array = lib.map_from_sym(fullsigns, symlabels, qtot=qtot, mod=mod)
           ans   = tn.Map(array, legs)

           sym = tn.Symmetry1D(fullsigns, symlabels, qtot=qtot, mod=mod) 
           out = tn.Map.compute(sym, legs)

           self.assertEqualMap(out, ans)
           self.assertMapVsArray(out, legs, array)
           self.assertMap(out, legs, sym.shape)

       Si = np.arange(0,4)
       Sj = np.arange(1,3)
       Sk = np.arange(2,5)
       Sl = np.arange(0,2)
       Sm = np.arange(0,3)
       Sn = np.arange(1,5)

       _test("IJKL", "--+-", [Si,Sj,Sk,Sl])
       _test("MLN",  "++-",  [Sm,Sl,Sn])
       _test("MLN",  "++-",  [Sm,Sl,Sn], qtot=2, mod=3)



   def test_contract_maps(self):

       def _test(A, B, legsC, shapeC):

           out   = tn.contract_maps(A, B, legsC)  
           out1  = A.contract(B, legsC)

           arrayC = np.einsum(subscript, A.array, B.array)
           ans    = tn.Map(arrayC, legsC)

           self.assertEqualMap(out,  ans)
           self.assertEqualMap(out1, ans)

           self.assertMapVsArray(out,  legsC, arrayC)
           self.assertMapVsArray(out1, legsC, arrayC)

           self.assertMap(out,  legsC, shapeC)
           self.assertMap(out1, legsC, shapeC)

       np.random.seed(1)

       A = lib.create_random_map("IJK",  (2,3,4),   5)
       B = lib.create_random_map("MI",   (3,2),     2)
       C = lib.create_random_map("UJIW", (5,3,2,5), 10)

       _test(A, B, "JKM",  (3,4,3))
       _test(A, C, "IKUW", (2,4,5,5))
       _test(C, A, "JUWK", (3,5,5,4))






















































































































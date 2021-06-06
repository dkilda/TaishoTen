#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import helper_lib as lib
import taishoten as tn

from taishoten import Str
from .util     import TaishoTenTestCase, must_fail




class TestTensor(TaishoTenTestCase):

   def test_construct(self):

       def _test(signs, symlabels, dense_shape):

           args = lib.setup_tensor(signs, symlabels, dense_shape)
           self.assertTensorVsArray(*args)

       np.random.seed(1)

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       _test("++-",    (Si,Sj,Sk),    (8,8,8))
       _test("++--",   (Sn,So,Sj,Sl), (8,8,8,8))
       _test("+0+0+-", (Si,Sn,So,Sm), (8,3,8,3,8,8))
       _test("++00-",  (Si,Sn,Sp),    (8,8,3,3,8))
       _test("++-0-",  (Sj,Sj,Sj,Sj), (5,5,8,3,8))

       array  = lib.randn(8,8,8,8)
       tensor = tn.Tensor(array)
       self.assertTensorVsArray(tensor, array)



   def test_as_new_tensor(self):

       np.random.seed(1)

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       A, arrayA, symA = lib.setup_tensor("++-",    (Si,Sj,Sk),    (8,8,8))
       B, arrayB, symB = lib.setup_tensor("++--",   (Sn,So,Sj,Sl), (8,8,8,8))
       C, arrayC, symC = lib.setup_tensor("+0+0+-", (Si,Sn,So,Sm), (8,3,8,3,8,8))

       out = A.as_new_tensor()
       self.assertEqualTensor(out, A)

       out = A.as_new_tensor(arrayB, symB)
       self.assertEqualTensor(out, B)

       _failed_test = must_fail(lambda x: B.as_new_tensor(sym=x))
       _failed_test(symC)



   def test_get_full_array(self):

       def _test(subscript, signs, symlabels, dense_shape):

           # Setup tensor
           tensor, array, sym = lib.setup_tensor(signs, symlabels, dense_shape)

           # Compute ans
           map_symlegs = tn.subscript_to_legs(subscript)[1]

           mp  = tn.Map.compute(sym.unset_map(), map_symlegs)
           ans = np.einsum(subscript, array, mp.array)

           # Compute out
           out = tensor.get_full_array()
           self.assertCloseArray(out, ans)


       # Define symlabels
       np.random.seed(1)

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       # Test-1
       sub = "IJijk,IJK->IJKijk"
       _test(sub, "++-", (Si,Sj,Sk), (8,8,8))

       # Test-2
       sub = "INOixnyom,INOM->INOMixnyom"
       _test(sub, "+0+0+-", (Si,Sn,So,Sm), (8,3,8,3,8,8))



   def test_get_symmetrized_array(self):

       def _test(subscript, signs, symlabels, dense_shape):

           # Make random full-size array
           sym_shape  = (len(s) for s in symlabels)
           full_array = lib.randn(*sym_shape, *dense_shape)

           # Make tensor
           tensor, array, sym = lib.setup_tensor(signs, symlabels, dense_shape)

           # Compute ans
           map_symlegs = tn.subscript_to_legs(subscript)[1]

           mp  = tn.Map.compute(sym.unset_map(), map_symlegs)
           ans = np.einsum(subscript, full_array, mp.array)

           # Compute out
           out  = tensor.get_symmetrized_array(full_array)
           out1 = tensor.get_symmetrized_array(array)

           self.assertCloseArray(out,  ans)
           self.assertCloseArray(out1, array)


       # Define symlabels
       np.random.seed(1)

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       # Test-1
       sub = "IJKijk,IJK->IJijk"
       _test(sub, "++-", (Si,Sj,Sk), (8,8,8))

       # Test-2
       sub = "INOMixnyom,INOM->INOixnyom"
       _test(sub, "+0+0+-", (Si,Sn,So,Sm), (8,3,8,3,8,8))
       


   def test_transform(self):

       def _test(tensor, path_key, *subs):

           # Get path
           trans = lib.TransformPathMaker()
           path  = trans.pathlets[path_key]

           # Compute ans
           array = tensor.array
           for i, sub in enumerate(subs):
               array = np.einsum(sub, array, path[i].map_array)

           ans = tn.Tensor(array, tensor.sym)

           # Compute out
           out  = tn.transform(tensor, path)
           out1 = tensor.transform(path)

           self.assertEqualTensor(out,  ans)
           self.assertEqualTensor(out1, ans)
                

       # Define symlabels
       np.random.seed(1)

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       # Test (NOJ -> JNQ -> LNQ)
       sub1 = "NOJnojl,NOQ->JNQnojl"
       sub2 = "JNQnojl,JLQ->LNQnojl"

       B = lib.setup_tensor("++--", (Sn,So,Sj,Sl), (8,8,8,8))
       _test(B, ("B", Str("LNQ")), sub1, sub2) 


      

























































































































































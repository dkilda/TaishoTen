#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import copy  as cp
import numpy as np

import lib
import util

import taishoten as tn
from taishoten import Str





class TestTensor:

   @pytest.fixture(autouse=True)
   def request_tensors(self, fixt_tensors_1D):
       self._tensors_and_data = fixt_tensors_1D


   @pytest.fixture(autouse=True)
   def request_transform_data(self, fixt_transform_data):
       self.trans = fixt_transform_data


   def tensors_and_data(self, key):
       tensor, data = self._tensors_and_data[key]
       return tensor, data


   def tensors(self, key):
       tensor, _ = self._tensors_and_data[key]
       return tensor





   # --- Test construction -------------------------------------------------- #

   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_construct(self, key):

       tensor, data = self.tensors_and_data(key)
       array        = data["array"]
       sym          = data["sym"]

       out = tn.Tensor(array, sym)
       util.assert_tensor_vs_array(out, array, sym)



   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_create(self, key):

       out, data = self.tensors_and_data(key)
       util.assert_tensor_vs_array(out, data["array"], data["sym"])




   @pytest.mark.parametrize("key, key1",           \
   [                                               \
   ["(2,2,2,2)",           "(2,2,2,2),nojl,++--"], \
   ["(2,2,2),ijk,++-",     "(2,2,2,2),nojl,++--"], \
   ["(2,2,2,2),inom,+++-", "(2,2,2,2),nojl,++--"], \
   ])
   @pytest.mark.xfail
   def test_create_failed(self, key, key1):

       tensor,  data  = self.tensors_and_data(key)
       tensor1, data1 = self.tensors_and_data(key1)

       out = tn.Tensor.create(data["array"], data1["sym"])
        



   @pytest.mark.parametrize("key, key1",                   \
   [                                                       \
   ["(2,2,2,2)",           "(2,2,2),ijk,++-"],             \
   ["(2,2,2),ijk,++-",     "(2,2,2,2),nojl,++--"],         \
   ["(2,2,2,2),nojl,++--", "(2,2,2,2),inom,+++-"],         \
   ["(2,2,2,2),nojl,++--", "(2,3,2,3,2,2),ixnyom,+0+0+-"], \
   ])
   def test_as_new(self, key, key1):

       tensor,  data  = self.tensors_and_data(key)
       tensor1, data1 = self.tensors_and_data(key1)

       out = tensor.as_new()
       util.assert_tensor_equal(out, tensor)

       out = tensor.as_new(array=data1["array"], sym=data1["sym"])
       util.assert_tensor_equal(out, tensor1)




   # --- Test get methods --------------------------------------------------- #


   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_get_dense_dims(self, key):

       tensor, data = self.tensors_and_data(key)
       array        = data["array"]
       legs         = data["legs"]
       dims         = array.shape[-len(legs) : ]
       
       out = tensor.get_dense_dims(legs)
       ans = dict(zip(legs, dims))
       assert out == ans




   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_get_map(self, key):

       tensor, data = self.tensors_and_data(key)
       sym          = data["sym"]
       symlegs      = data["symlegs"] 

       out = tensor.get_map(symlegs)

       if   sym is None:
            assert out is None
       else:
            ans = tn.Map.compute(sym, symlegs) 
            util.assert_map_equal(out, ans)
       
       


   # --- Test auxiliary methods --------------------------------------------- #

   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \

   ])
   def test_symmetrize(self, key):

       tensor, data = self.tensors_and_data(key)
       array        = cp.deepcopy(data["array"])
       sym          = data["sym"]
       legs         = data["legs"]
       symlegs      = data["symlegs"]       

       # De-symmetrize array
       mp        = tn.Map.compute(sym, symlegs)
       subscript = tn.util.legs_to_subscript(mp.legs, mp.legs, mp.legs[:-1])
       mp2_array = np.einsum(subscript, mp.array, mp.array)

       idx = lib.find_zeros(mp2_array)
       idx = [np.unravel_index(i, mp2_array.shape) for i in idx]
       idx = sorted(set(idx))

       for i in idx:
           array[i] = array[i] + 1

       # Symmetrize
       out = tn.Tensor(array, sym)
       out.symmetrize()

       # Test
       util.assert_tensor_equal(out, tensor)



   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_get_full_array(self, key):

       tensor, data = self.tensors_and_data(key)
       array        = data["array"]
       sym          = data["sym"]
       legs         = data["legs"]
       symlegs      = data["symlegs"] 

       # Compute ans
       if   sym is None:
            ans = array

       else:
            mp = tn.Map.compute(sym, symlegs)

            subscript_legs = [symlegs[:-1] + legs, symlegs, symlegs + legs]
            subscript      = tn.util.legs_to_subscript(*subscript_legs)

            ans = np.einsum(subscript, array, mp.array)

       # Compute out
       out = tensor.get_full_array()

       # Test
       util.assert_array_close(out, ans)





   # --- Test transformation ------------------------------------------------ #

   @pytest.mark.parametrize("key, path_key, subs",                       \
   [                                                                     \
   ["(2,2,2,2),ijlm,+++-",  None,             []],                       \
   ["(2,2,2,2),ijlm,+++-", ("A", Str("JMQ")), ["IJLijlm,IJLM->IJMijlm",  \
                                               "IJMijlm,IMQ->JMQijlm"]], \
   ])
   def test_transform(self, key, path_key, subs):

       tensor = self.tensors(key)
       path   = self.trans.pathlets[path_key] if path_key is not None else []
       array  = tensor.array
       sym    = tensor.sym

       # Compute ans
       for i, sub in enumerate(subs):
           array = np.einsum(sub, array, path[i].map_array)

       ans = tn.Tensor(array, sym)

       # Compute out
       out  = tn.tensor.transform(tensor, path)
       out1 = tensor.transform(path)

       # Test
       util.assert_tensor_equal(out,  ans)
       util.assert_tensor_equal(out1, ans)




   @pytest.mark.parametrize("key, key_fw, subs_fw, key_bw, subs_bw",       \
   [                                                                       \
   ["(2,2,2,2),inom,+++-", ("C1", Str("MOQ")), ["INOinom,IMNO->IMOinom",   \
                                                "IMOinom,IMQ->MOQinom"],   \
                           ("C",  Str("MOQ")), ["MOQinom,IMQ->IMOinom",    \
                                                "IMOinom,IMNO->INOinom"]], \
   ])
   def test_transform_reverse(self, key, key_fw, subs_fw, \
                                         key_bw, subs_bw  ):

       tensor  = self.tensors(key)
       path_fw = self.trans.pathlets[key_fw] 
       path_bw = self.trans.pathlets[key_bw] 
       array   = tensor.array
       sym     = tensor.sym

       # Forward transformation
       for i, sub in enumerate(subs_fw):
           array = np.einsum(sub, array, path_fw[i].map_array)
 
       tensor = tn.Tensor(array, sym)

       # Compute ans
       for i, sub in enumerate(subs_bw):
           array = np.einsum(sub, array, path_bw[i].map_array)

       ans = tn.Tensor(array, sym)

       # Compute out
       out  = tn.tensor.transform(tensor, path_bw)
       out1 = tensor.transform(path_bw)

       # Test
       util.assert_tensor_equal(out,  ans)
       util.assert_tensor_equal(out1, ans)









































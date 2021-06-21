#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import copy  as cp
import numpy as np

import lib
import util
import taishoten as tn



def random_put(shape, n=0.5):

    high = np.prod(shape)

    if   type(n) is int:
         size = n
    else: 
         size = round(n*high)

    np.random.seed(1)
    idx  = lib.randint(high=high, size=size)
    idx  = np.unique(idx)
    vals = 1.0

    A = np.zeros(shape)
    A.put(idx, vals)

    return A, idx, vals



@pytest.fixture
def ndarray_fixt():

    dct = {}
    dct["A"] = lib.randn(2,3,4)
    dct["B"] = lib.randn(5,2,5,3)
    return dct



class TestBackend:

   @pytest.mark.parametrize("x", \
   [None, "numpy", tn.backends.numpy.NumpyBackend()])
   def test_get_backend(self, x):

       out = tn.backends.get_backend(None)
       ans = tn.backends.numpy.NumpyBackend()

       assert type(out) == type(ans)


   @pytest.mark.parametrize("x", [12, "tensorflow"])
   @pytest.mark.xfail
   def test_get_backend_failed(self, x):

       out = tn.backends.get_backend(x)




class TestNumpyBackend:


   def prepare(self, ndarray_fixt, key):

       backend = tn.backends.numpy.NumpyBackend()
       x = ndarray_fixt[key]

       return x, backend


   @pytest.mark.parametrize("key", ["A", "B"])
   def test_asarray(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)

       out = backend.asarray(x)
       util.assert_array_close(out, x)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_zeros(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)

       out = backend.zeros(x.shape)
       ans = np.zeros(x.shape)

       util.assert_array_close(out, ans)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_ones(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)

       out = backend.ones(x.shape)
       ans = np.ones(x.shape)

       util.assert_array_close(out, ans)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_random(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)

       np.random.seed(1)
       out = backend.random(x.shape)

       np.random.seed(1)
       ans = np.random.random(x.shape)

       util.assert_array_close(out, ans)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_eye(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)

       out = backend.eye(x.shape[0])
       ans = np.eye(x.shape[0])

       util.assert_array_close(out, ans)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_copy(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)

       out = backend.copy(x)
       util.assert_array_close(out, x)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_shape(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)

       out = backend.shape(x)
       assert out == x.shape



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_put(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)
     
       ans, idx, vals = random_put(x.shape)

       out = np.zeros(x.shape)
       out = backend.put(out, idx, vals)

       util.assert_array_close(out, ans)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_find_zeros(self, ndarray_fixt, key, tol=1e-6):

       x, backend  = self.prepare(ndarray_fixt, key)
       arr, idx, _ = random_put(x.shape)

       out  = backend.find_zeros(arr, tol)
       ans  = np.where(abs(arr.ravel()) < tol)[0]
       ans1 = [i for i in range(x.size) if i not in idx]

       util.assert_array_close(out, ans)
       util.assert_list(out, ans1)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_find_nonzeros(self, ndarray_fixt, key, tol=1e-6):

       x, backend  = self.prepare(ndarray_fixt, key)
       arr, idx, _ = random_put(x.shape)

       out  = backend.find_nonzeros(arr, tol)
       ans  = np.where(abs(arr.ravel()) > tol)[0]
       ans1 = sorted(idx) 

       util.assert_array_close(out, ans)
       util.assert_list(out, ans1)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_norm(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)

       out  = backend.norm(x)
       ans  = np.linalg.norm(x)

       util.assert_array_close(out, ans)



   @pytest.mark.parametrize("key", ["A", "B"])
   def test_dot(self, ndarray_fixt, key):

       x, backend = self.prepare(ndarray_fixt, key)

       a = x
       b = np.swapaxes(x, -1, -2)

       out  = backend.dot(a,b)
       ans  = np.dot(a,b)

       util.assert_array_close(out, ans)



   @pytest.mark.parametrize("key, subscript", [["A", "ijk,mjn->ijmn"], \
                                               ["B", "ijkl,kmin->jlmn"]])
   def test_einsum(self, ndarray_fixt, key, subscript):

       x, backend = self.prepare(ndarray_fixt, key)

       out = backend.einsum(subscript, x, x)
       ans = np.einsum(subscript, x, x)

       util.assert_array_close(out, ans)





































































































































#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools

from .backend import BaseBackend 


class NumpyBackend(BaseBackend):

  def __init__(self):
      self._name = "numpy"


  # --- Array creation methods ------------------------------------------------

  def asarray(self, array, *args, **kwargs):
      return np.asarray(array, *args, **kwargs)

  def zeros(self, shape):
      return np.zeros(shape)

  def ones(self, shape):
      return np.ones(shape)

  def random(self, shape):
      return np.random.random(shape)

  def eye(self, dim):
      return np.eye(dim)

  def copy(self, array):
      return array.copy()


  # --- Array properties and setting/manipulating values ----------------------

  def shape(self, array):
      return array.shape

  def put(self, array, idx, vals):
      array.put(idx, vals)
      return array

  def find_zeros(self, array, tol):
      idx = np.where(abs(array.ravel()) < tol)[0]
      return idx

  def find_nonzeros(self, array, tol):
      idx = np.where(abs(array.ravel()) > tol)[0]
      return idx


  # --- Calculations ----------------------------------------------------------
  
  def norm(self, *args, **kwargs):
      return np.norm(*args, **kwargs)

  def dot(self, *args, **kwargs):
      return np.dot(*args, **kwargs)

  def einsum(self, *args, optimize=True, **kwargs):
      return np.einsum(*args, **kwargs, optimize=optimize) 




























































































































































































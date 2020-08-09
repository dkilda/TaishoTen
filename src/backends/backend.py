#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy  as cp
import itertools


class BaseBackend:

  def __init__(self):
      self._name = "basebackend"


  # --- Array creation methods ------------------------------------------------

  def asarray(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented asarray".format(self._name))

  def zeros(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented zeros".format(self._name))

  def ones(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented ones".format(self._name))

  def random(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented random".format(self._name))

  def eye(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented eye".format(self._name))

  def copy(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented copy".format(self._name))

 
  # --- Array properties and setting/manipulating values ----------------------

  def shape(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented shape".format(self._name))

  def put(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented put".format(self._name))

  def find_zeros(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented find_zeros".format(self._name))

  def find_nonzeros(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented find_nonzeros".format(self._name))


  # --- Calculations ----------------------------------------------------------
  
  def norm(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented norm".format(self._name))

  def dot(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented dot".format(self._name))

  def einsum(self):
      raise NotImplementedError(\
            "Backend '{}' has not implemented einsum".format(self._name))





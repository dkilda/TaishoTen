#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy  as cp
import itertools

from .backend import BaseBackend
from .numpy   import NumpyBackend



class BackendDepot:

  def __init__(self):

      # Default backend
      self._default_backend = "numpy"

      # Backends and instantiated backends 
      self._BACKENDS = {"numpy": NumpyBackend}
      self._INSTANTIATED_BACKENDS = {} 


  def get_backend(self, backend):

      # Default backend
      if  backend is None:
          backend = self._default_backend

      # If backend is a BaseBackend child object: just return it
      if  isinstance(backend, BaseBackend):
          return backend

      # Otherwise, make sure backend input is a string
      if  not isinstance(backend, str): 
          raise ValueError("backend '{}' must be \
                           either a string or a BaseBackend child object".format(backend))

      # Make sure backend exists
      if  backend not in self._BACKENDS:
          raise ValueError("Backend '{}' does not exist".format(backend))

      # Instantiate backend if it's not already (we instantiate each backend only once)
      if  backend not in self._INSTANTIATED_BACKENDS:
          self._INSTANTIATED_BACKENDS[backend] = self._BACKENDS[backend]()
            
      # Return the instantiated backend object
      return self._INSTANTIATED_BACKENDS[backend]


# Create a backend depot object
# (must do it here so that it's not re-instantiated each time, 
#  which would defeat the purpose of _INSTANTIATED_BACKENDS)
backend_depot = BackendDepot()


# Get backend 
def get_backend(backend):  
    return backend_depot.get_backend(backend)





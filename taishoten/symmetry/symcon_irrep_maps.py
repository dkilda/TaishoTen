#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

import numpy as np
import copy  as cp
import itertools

from util import Str
from util import assert_equal, isiterable, del_from_list, multisorted
from util import NotImplementedField
from util import subscript_to_legs, legs_to_subscript
from util import symlegs_and_denselegs_to_subscript

from .symmetry    import SymFactory
from .symcon_base import SymCon



class SymConWithIrrepMaps(SymCon):

  def __init__(self, symA, symB, denselegs):

      # Create SymCon without maps first 
      super().__init__(symA, symB, denselegs)

      # List of maps and map legs
      self._maps    = None
      self._maplegs = None 

      # Symleg and denseleg dims 
      # (to be determined using maps and external info)
      self._denseleg_dims = None
      self._symleg_dims   = None


  # --- Get methods and properties --------------------------------------------------------------------------

  def get_maps_list(self):
      return self._maps

  def get_maplegs_list(self):
      return self._maplegs


  # --- Compute the list of maps and map legs ---------------------------------------------------------------

  def compute_maps(self): 

      # Build a full list of maps, including the auxiliary ones

      # Initialize
      self._maps    = []
      self._maplegs = []

      # Get delta maps, add them to the list
      mapA = self.symA.get_map()
      mapB = self.symB.get_map()

      self.add_map(mapA, self.symlegs["A"])
      self.add_map(mapB, self.symlegs["B"])

      # Contracted legs and remaining legs of A and B
      RES_A = self.symlegs["A"] - self.CON
      RES_B = self.symlegs["B"] - self.CON

      # If num of contracted and resulting legs is more than one, 
      # we'll need an auxiliary leg and an auxiliary map
      if  len(self.CON) > 1 and len(RES_A) > 1 and len(RES_B) > 1:

          # Compute auxiliary map and its legs
          aux_map     = self.compute_aux_map()
          aux_maplegs = self.CON + Str('Q')

          # Add them to the list
          self.add_map(aux_map, aux_maplegs)

      # Combine the input delta maps to make a full list of maps and legstrings
      # E.g. if we've map_legs = [IJLM, NOJL], we create an additional auxiliary map with aux_map_legstr = JLQ 
      #      --> so in the end we've map_legs = [IJLM, NOJL, JLQ]
      self._compute_contracted_maps()

      # Sort all maps by length of their maplegs
      self._maplegs, self._maps = multisorted(self._maplegs, self._maps, key=len)
      return self


  def _compute_contracted_maps(self):

      # We will iterate over all possible pairs of input maps, and create a combo map for each pair
      #
      # -- if input maps only contain two maps (corresponding to A,B) 
      #    then only one extra map (corresponding to C) will be produced,
      #       e.g. map_legs = [IJK, KLM] -> [IJK, KLM, IJLM] 
      #
      # -- if input maps also contain aux map, then more maps get produced
      #       e.g. map_legs = [IJLM, NOJL, JLQ] -> [IJLM, NOJL, IMNO, JLQ, IMQ, NOQ] 
      #

      # Num of maps before any contracted maps were added (we'll iterate only over non-contracted maps)
      # Create all possible pairings of maps
      num_maps = len(self._maps)
      idx_of_all_map_pairings = itertools.combinations(range(num_maps), 2) 

      # Loop over all possible pairs of maps in the input: we'll compute a combo map for each pair
      for i,j in idx_of_all_map_pairings:

          # For a pair of (i,j) maps, 
          # find contracted legs (CON), result legs (RES), all legs (ALL)
          ALL = self._maplegs[i] | self._maplegs[j]
          CON = self._maplegs[i] & self._maplegs[j]
          RES = ALL - CON 

          # If no CON legs: cannot contract the maps, skip to next iter
          if  len(CON) == 0: 
              continue

          # If no RES legs or RES legs give one of the existing maps: skip to next iter
          if  len(RES) == 0 or (RES in self._maplegs):
              continue

          # Contract (i,j) maps --> gives a new map  
          subscript = legs_to_subscript([self._maplegs[i], self._maplegs[j], RES])
          new_map   = self.contract_maps(subscript, self._maps[i], self._maps[j])

          # Add [RES legs (new map legs)] and [new map] to the list of maps
          self.add_map(new_map, RES) 

      return self


  def _compute_aux_map(self):

      # Compute auxiliary symmetry
      aux_sym = self._compute_aux_sym()

      # Compute auxiliary map
      aux_sym.compute_map()
      aux_map = aux_sym.get_map()
      return aux_map


  def _compute_aux_sym(self):

      # Get indices of CON legs for A and B tensors
      idx_CON_A = [self.symlegs["A"].find(i) for i in self.CON] 
      idx_CON_B = [self.symlegs["B"].find(i) for i in self.CON]

      # When computing symrange: 
      #      if [CON legs of tensor A] have the opposite sign than [CON legs of tensor B]:
      #      then [symlabels of B] will have the opposite sign than [symlabels of A]
      # Hence, there's a phase factor opposite to the relative phase calculated before.
      phase = -1 * self._phase 

      # Compute auxiliary symranges for A and B tensors
      aux_symrange_A = self.symA.make_aux_symrange(indices=idx_CON_A)
      aux_symrange_B = self.symB.make_aux_symrange(indices=idx_CON_B, phase=phase)

      # Merge them together into a single symrange
      aux_symrange = self.merge(aux_symrange_A, aux_symrange_B)

      # Sign-string of [the new aux map]: signs     of CON-A legs + '-ve' sign of aux leg
      # Symrange    of [the new aux map]: symranges of CON-A legs + aux symrange computed above
      #
      signs    = ''.join([self.symA.puresigns[i] for i in idx_CON_A]) + '-'
      symrange =         [self.symA.symrange[i]  for i in idx_CON_A]  + [aux_symrange]

      # Construct the symmetry defining [the new aux map]
      aux_sym = Symmetry(signs, symrange, qtot=0, mod=None, backend=self.backend)
      return aux_sym



  # --- Contracting maps ------------------------------------------------------------------------------------

  def contract_maps(subscript, mapA, mapB):
      
      # Contract maps A and B, 
      # find indices of nonzero entries in the resulting array
      res   = self.backend.einsum(subscript, mapA, mapB)
      shape = self.backend.shape(res)
      idx   = self.backend.find_nonzeros(res)

      # Create map C
      mapC = self.generate_map_from_indices(shape, idx)
      return mapC  


  def generate_map_from_indices(self, shape, idx):  
  
      # Generate map from symrange indices: use method from either symA or symB
      return self.symA.generate_map_from_indices(shape, idx)



  # --- Determine leg dims ----------------------------------------------------------------------------------

  def make_denseleg_dims(self, dense_subscript, dense_shape_A, dense_shape_B): 

      # Dense shape
      shape_AB = shape_A + shape_B

      # Dense leg
      legs_tmp = subscript_to_legs(dense_subscript)
      legs_A   = legs_tmp["A"].get_str()
      legs_B   = legs_tmp["B"].get_str()
      legs_AB  = legs_A + legs_B
    
      # Make dense dims dict
      self._denseleg_dims = {leg: dim for leg, dim in zip(legs_AB, shape_AB)}
      return self
        
         
  def make_symleg_dims(self): 

      # Get all map legs and their dims
      symlegs_list = [leg for maplegs in self._maplegs for leg in maplegs.get_str()]    
      mapdims_list = [dim for mp      in self._maps    for dim in mp.shape()]    
        
      # Make sym dims dict
      self._symleg_dims = {leg: dim for leg, dim in zip(symlegs_list, mapdims_list)}
      return self


  # ---------------------------------------------------------------------------------------------------------


























































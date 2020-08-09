#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools

from .util import Str
from .util import assert_equal, isiterable, del_from_list, multisorted
from .util import subscript_to_legs, legs_to_subscript
from .util import symlegs_and_denselegs_to_subscript


# --- Find the optimal transformation sequence for A,B,C tensors, using the symcon depot input --------------

def find_optimal_transform_sequence(symcon):

    # Unpack SymCon depot: 
    denselegs     = symcon.get_denselegs()
    denseleg_dims = symcon.get_denseleg_dims() 
    symlegs       = symcon.get_symlegs()
    symleg_dims   = symcon.get_symleg_dims()
    maplegs_list  = symcon.get_maplegs_list()
    maps_list     = symcon.get_maps_list()

    # Build all configs for (A,B,C)
    configweb_ABC, configs_ABC = make_configweb_ABC(symlegs, maplegs_list, maps_list)

    # Estimate memory cost associated with dense legs, get num of independent symlegs
    dense_mem_cost  = estimate_mem_cost_ABC(denselegs, denseleg_dims)
    num_ind_symlegs = get_num_ind_symlegs(symlegs)

    # Initialize lists for storing transform sequences, 
    # legs of final configs in each sequence, mem cost of each sequence
    transform_sequences_list = []
    final_legs_list          = []
    mem_cost_list            = []

    # Compute cartesian product of A,B,C configs -- loop over all combos of A,B,C configs (a,b,c)
    cartesian_prod = itertools.product(configs_ABC["A"], \
                                       configs_ABC["B"], \
                                       configs_ABC["C"])
    for a, b, c in cartesian_prod:

        # Get final symleg configs for A,B,C tensors
        final_config_ABC      = {"A": a, "B": b, "C": c}  
        final_config_legs_ABC = make_final_config_legs_ABC(final_config_ABC)
    
        # Verify if symlegs of final configs (a,b,c) can be contracted directly: 
        # if not, skip to next combo of final configs
        if  not good_to_contract(final_config_legs_ABC, num_ind_symlegs):
            continue

        # Compute transform sequences for (A,B,C) and their total memory cost
        transform_sequence_ABC, cost = make_transform_sequence_ABC(configweb_ABC, final_config_ABC, \
                                                                   dense_mem_cost, symleg_dims)
        # Add to list
        transform_sequences_list.append(transform_sequence)
        final_legs_list.append(final_config_legs)
        cost_list.append(cost)

    # Find the optimal (minimum memory cost) transform sequence
    idx                            = cost_list.index(min(cost_list))
    optimal_transform_sequence_ABC = transform_sequences_list[idx]

    # Make contraction subscript 
    final_symlegs_ABC = final_legs_list[idx]
    subscript         = symlegs_and_denselegs_to_subscript(final_symlegs_ABC, denselegs)

    return optimal_transform_sequence_ABC, subscript



def make_configweb_ABC(input_symlegs, maplegs_list, maps_list):

    # Build config web and get list of configs for each of (A,B,C)
    configweb = {}
    configs   = {}

    for key in input_symlegs.keys():
        configweb[key] = ConfigWeb(input_symlegs[key], maplegs_list, map_list)
        configs        = configweb[key].get_configs()  

    return configweb, configs



def estimate_mem_cost_ABC(legs, legdims):

    # Estimate memory cost for (A,B,C)
    mem_cost = {}
    for key in legs.keys():
        mem_cost[key] = estimate_mem_cost(legs[key], legdims)
    return mem_cost



def make_final_config_legs_ABC(final_config):

    # Get legs of each (A,B,C) final config 
    final_config_legs = {}
    for key in final_config.keys():
        final_config_legs[key] = final_config[key].legs
    return final_config_legs


    
def make_transform_sequence_ABC(configweb, final_config, dense_mem_cost, symleg_dims):

    # Compute transform sequences and their memory costs for each of (A,B,C)
    transform_sequences = {}
    mem_costs           = {}

    for key in configwebs.keys():

        # Get config and dense mem cost for each of (A,B,C), reverse "C" transform sequence
        config    = final_config[key]
        dense_mem = dense_mem_cost[key]
        reverse   = (key == "C")

        # Compute a sequence of transforms for each of (A,B,C), get its memory cost
        transform_sequences[key] = configweb[key].compute_transform_sequence(config, reverse)
        mem_costs[key]           = configweb[key].estimate_mem_cost(config, dense_mem, symleg_dims)

    # Get the total memory cost (sum over costs of (A,B,C))
    total_cost = sum(mem_costs.values())
    return transform_sequences, total_cost

        


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################



class ConfigWeb:

  def __init__(self, input_symlegs, maplegs_list, map_list):

      # Initialize config list by adding a [starting node] corresponding to the [input leg configuration]
      self._config_list = []
      self.add_config(input_symlegs, 0, None)

      # Initialize transform sequence
      self._transform_sequence = []
      self._mem_cost = None

      # List of map legs corresponding to maps that we'll apply 
      self._maplegs_list = maplegs_list
      self._map_list = map_list

      # Num of independent output symlegs
      self._num_output_symlegs = self.get_num_ind_symlegs_after_transform_sequence(input_symlegs) 



  # --- Get methods and properties: configs and transforms --------------------------------------------------

  def get_configs(self):
      return cp.deepcopy(self._configs_list)


  def get_config(self, idx): 
      return cp.deepcopy(self._configs_list[idx])


  def get_config_legs(self, idx=None):
      if   idx is None:
           config_legs_list = [config.legs for config in self._config_list]
           return config_legs_list
      else:
           return self._config_list[idx].legs


  def get_num_ind_symlegs_after_transform_sequence(self, input_symlegs):

      # There's one dependent leg per [maplegs group with unseen legs], 
      #         i.e. legs that are not subset of input legs:
      #              count the number of [mapleg groups] that have unseen legs
      num_dep_legs = 0 
      for maplegs in self._maplegs_list:
          if  maplegs.issubset(input_symlegs)
              num_dep_legs += 1
 
      # Get num of independent symlegs in the output symlegs string after a sequence of transformations 
      # NB. [num of ind output legs] = [num of ind input legs], 
      #     b/c each map can only replace one leg by another: num of ind legs remains the same during any transformation
      num_legs     = len(input_symlegs)
      num_ind_legs = num_legs - num_dep_legs 
      return num_ind_legs 


  @property
  def num_configs(self):
      return len(self._config_list)


  @property
  def num_transforms(self):
      return len(self._transform_sequence)



  # --- Add configs and transforms --------------------------------------------------------------------------

  def add_config(self, legs, previous_idx, map_idx):

      # Create and add a new config to the list
      config = Config(legs, previous_idx, map_idx)
      self._config_list.append(config)
      return self


  def add_transform(self, start_legs, map_legs, end_legs, irrep_map):

      # Create and add to the list a new transform between two different configs
      transform = Transform(start_legs, map_legs, end_legs, irrep_map)
      self._transform_list.append(transform)
      return self



  # --- Build a web of configurations -----------------------------------------------------------------------

  def build(self):

      # Build the first web strand with root node at idx = 0
      self._build_web_strand(0)

      # Num of parent configs = [num of configs in the first layer] - [1 = starting config]
      num_roots = self.num_configs - 1
 
      # Build other layers: use the [already added output legs] as new starting pts
      for root_idx in range(1, num_roots + 1):
          self._build_web_strand(root_idx)

      return self



  def _build_web_strand(self, root_idx): 

      # Take the root config of the current web strand, use it as a starting point for transformations
      legs = self.get_config_legs(root_idx)  

      for map_idx, map_legs in enumerate(self._maplegs_list):

          # If the difference between [map legs] and [input legs] is more than one leg: 
          #    no valid transformation can be done, skip to next iteration 
          #    (b/c map only adds && removes one leg, cannot deal with two-leg differences)
          if  len(map_legs - legs) > 1: 
              continue

          # Get [all legs], [shared legs], [extra legs]
          all_legs    = legs | map_legs
          shared_legs = legs & map_legs
          extra_legs  = all_legs - shared_legs

          # If [num of extra/unshared legs] is more than [num of independent legs]: 
          #    no valid transformation can be done, skip to next iteration
          #    otherwise [output legs] will contain dependent legs 
          #    b/c [output legs] -> [shared legs subgroup] + [extra legs]
          if  len(extra_legs) > self._num_output_symlegs: # FIXME could set extra_legs = legs - map_legs
              continue

          # Form all possible output combos: [output legs] = [subgroup of shared legs] + [extra/unshared legs] 
          self._build_configs_for_this_map(map_idx, root_idx, map_legs, all_legs, shared_legs, extra_legs)



  def _build_configs_for_this_map(self, map_idx, root_idx, map_legs, all_legs, shared_legs, extra_legs):

      # Form all possible output combos: [output legs] = [subgroup of shared legs] + [extra/unshared legs] 
      shared_legs_str    = shared_legs.get_str()
      shared_legs_combos = itertools.combinations(shared_legs_str, self._num_output_symlegs - len(extra_legs))

      # Loop over all [subgroups of shared legs], each group of size |num_out - num extra legs|
      for shared_legs_subset in shared_legs_combos:

          # Make [output legs] = [subgroup of shared legs] + [extra/unshared legs]
          output_legs = Str(shared_legs_subset) + extra_legs
          other_legs  = all_legs - output_legs
          config_legs = self.get_config_legs()

          # Check if [output legs] is a valid combo. Reasons to reject an output leg combo:
          #
          # -- if [output legs] in [config node legs]: we've already considered this output leg combo
          #
          # -- if [output legs] in [map legs set]: all [output legs] are contained in one of the maps, 
          #       thus [output legs] contains some dependent legs
          #
          # -- if |[all legs] - [output legs]| >= 2: [output legs] 
          #       is missing some independent legs and thus is not valid
          #       i.e. [all legs] contains up to 1 dependent leg so |[all legs] - [output legs]| < 2 must hold
          #
          if  output_legs not in config_legs and \
              output_legs not in map_legs    and len(other_legs) < 2:

              # Add new configuration to config_list
              self.add_config(output_legs, root_idx, map_idx)



  # --- Compute a sequence of transforms linking different configs in the ConfigWeb -------------------------

  def compute_transform_sequence(self, final_config, reverse=False):

      # Init empty transform sequence
      self._transform_sequence = []

      # Set current config = final/target config of the transform sequence 
      # (then we'll travel down config tree backwargs)
      current_config = final_config

      while current_config.map_idx is not None:
           
            # Get previous config
            previous_config = self._config_list[current_config.previous_idx]

            # Get start, end, map legs -- and the map operator itself
            start_legs = previous_config.legs
            end_legs   = current_config.legs
            map_legs   = self._maplegs_list[current_config.map_idx]  
            irrep_map  = self._map_list[current_config.map_idx]        

            # Follow the link to previous transformation
            # (keep doing this until we hit the first node with map_idx = None)
            current_config = previous_config

            # Add a new transform node to the transformation sequence
            if  reverse:
                self.add_transform(end_legs,   map_legs,  start_legs, irrep_map)
            else:
                self.add_transform(start_legs, map_legs,  end_legs,   irrep_map)

      # Reverse the sequence to get start-to-end order 
      # (unless we mean to keep it reversed in end-to-start order, i.e. reverse=True)
      if  not reverse:
          self._transform_sequence = self._transform_sequence[::-1]
      return self._transform_sequence



  def estimate_mem_cost(self, final_config, dense_mem_cost, symleg_dims):

      # Estimate memory cost associated with symlegs in the final/target config of the transform sequence 
      # --> equal to the product of all non-hidden symleg dims in the final config 
      #     (i.e. mem cost of storing the final config)
      symlegs      = final_config.legs 
      sym_mem_cost = estimate_mem_cost(symlegs, symleg_dims)

      # Get the total memory cost of this transform sequence: 
      # --> multiply [denselegs mem cost] * [symlegs mem cost] * [num of steps in transform sequence] 
      self._mem_cost = dense_mem_cost * sym_mem_cost * (self.num_transforms - 1)          
      return self._mem_cost




#######################################################################################################################
#######################################################################################################################
#######################################################################################################################



# --- Config data-container class ------------------------------------------------------------------------------------- 

class Config:

  def __init__(self, legs, previous_idx, map_idx=None):

      self._config_legs            = legs
      self._previous_config_idx    = previous_idx
      self._map_to_this_config_idx = map_idx

  @property
  def legs(self):
      return cp.deepcopy(self._config_legs) 

  @property
  def previous_idx(self):
      return cp.deepcopy(self._previous_config_idx) 

  @property
  def map_idx(self):
      return cp.deepcopy(self._map_to_this_config_idx)




# --- Transform data-container class ----------------------------------------------------------------------------------

class Transform:

  def __init__(self, start_legs, map_legs, end_legs, irrep_map):

      self._start_legs = start_legs
      self._map_legs   = map_legs
      self._end_legs   = end_legs
      self._map        = irrep_map

  @property
  def start_legs(self):
      return cp.deepcopy(self._start_legs) 

  @property
  def map_legs(self):
      return cp.deepcopy(self._map_legs) 

  @property
  def end_legs(self):
      return cp.deepcopy(self._end_legs) 

  def get_map(self):
      return cp.deepcopy(self._map)




#######################################################################################################################
#######################################################################################################################
#######################################################################################################################





# --- Auxiliary functions -----------------------------------------------------------------------------------


def good_to_contract(symlegs, num_ind_symlegs=None):

    # Set num of independent legs
    if  num_ind_symlegs is None:
        num_ind_symlegs = get_num_ind_symlegs(symlegs)

    # Truncate symlegs (if not truncated already)
    symlegs = {key: symlegs[key].truncate() for key in symlegs.keys()}

    # Input and output legs
    legs_in  = symlegs["A"] + symlegs["B"]
    legs_out = symlegs["C"]  

    # Two conditions:
    # -- [num input legs] = [num delta-independent legs]: input contains no delta-dependent legs, 
    #                                                     and doesn't omit any delta-independent legs 
    # -- [output legs] in [input legs]: no new legs (not seen in the input) appear in the output
    #
    is_contractable = (len(legs_in) == num_ind_legs and legs_out.issubset(legs_in))
    return is_contractable



def get_num_ind_symlegs(symlegs):

    # Sort leg groups by length: must iterate over leg groups from shortest-to-longest, 
    # to count the number of dependent legs correctly
    symlegs_list        = symlegs.values()
    sorted_symlegs_list = sorted(symlegs_list, key=len)  
      
    # Count the number of delta-dependent legs  
    num_dep_legs = 0
    visited_legs = Str("")

    for legs in sorted_symlegs_list:

        # There's one dependent leg per [leg group with unvisited legs]: 
        #         count the number of [leg groups] that have unvisited legs
        if  not legs.issubset(visited_legs):
            visited_legs  = visited_legs | legs
            num_dep_legs += 1

    # Get the number of independent legs
    num_legs     = get_num_legs(symlegs_list)  
    num_ind_legs = num_legs - num_dep_legs           
    return num_ind_legs



def estimate_mem_cost(legs, legdims_dict):

    # Multiply dims of all legs involved
    cost = 1
    for leg in legs.get_str():
        cost *= legdims_dict[leg]

    return cost

































































































































































































































































































































































































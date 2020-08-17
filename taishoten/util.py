#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools



class Str:

  def __init__(self, legs, truncated=False, order=None):

      # Create String
      self._initialize(legs, truncated, order)


  def __call__(self, legs, truncated=False, order=None):

      # Update String
      self._initialize(legs, truncated, order)
      return self


  def _initialize(self, legs, truncated=False, order=None):

      # Initialize string 

      # Make sure all legs are unique so that our String can be treated as str and set simultaneously
      assert_unique(legs, "Str: input legs must be unique")

      # Default order (only needed when your input is a set)    
      if  order is None:
          order = {leg: i for i, leg in enumerate(legs)}  # order = cp.copy(legs)

      # If input is a set, make sure legs are ordered correctly
      if  type(legs) == set: 
          keyfunc = lambda key: order[key]  # keyfunc = lambda key: order.find(key)
          legs    = ''.join(sorted(legs, key=keyfunc))

      # Only a string input allowed at this point
      assert_equal(type(legs), str, "Str: input legs must be str")

      # Set legs
      self._legs      = legs
      self._truncated = truncated
      self._order     = order


  # --- Get methods -----------------------------------------------------------------------------------------

  def get_str(self):
      return ''.join(self._legs)

  def get_set(self):
      return set(self._legs)

  def get_order(self):
      return self._legs
      

  # --- Uppercase/lowercase, truncation, order, etc ---------------------------------------------------------

  def uppercase(self):
      # Convert to uppercase
      self._legs.upper()
      return self

  def lowercase(self):
      # Convert to lowercase
      self._legs.lower()
      return self

  def purify(self, signs):
      # Purify from 0-signed legs
      legs = ''.join([leg for i, leg in enumerate(self._legs) if signs[i] != '0'])
      legs.upper()
      return self(legs)


  def truncate(self):

      # If already truncated, don't truncate anymore
      if  self.truncated:
          return self

      # Truncate the last leg
      trunc_legs = self._legs[:-1]
      return self(trunc_legs, truncated=True)


  def truncated(self):
      # Check if String() has been truncated
      return self._truncated


  def combine(self, other, new_legs):

      # Combine "self" and "other", 
      # given a set of new legs that "self" and "other" combination will contain
      new_order = self.combine_orders(other, new_legs)
      return self(new_legs, order=new_order)


  def combine_orders(self, other, new_legs):

      # Get leg orders in "self" and "other"
      self_order  = self.get_order()
      other_order = other.get_order()

      # Discard legs not present in "self" and "other" combination
      self_order  = [leg for leg in self_order  if leg in new_legs]
      other_order = [leg for leg in other_order if leg in new_legs]

      # Concatenate "self" and "other" orders
      new_order = self_order + other_order 
      return new_order


  # --- Override [] access and mutation magic methods -------------------------------------------------------

  def __getitem__(self, key):
      return self._legs[key]

  def __setitem__(self, key, val):
      self._legs[key] = val
      assert_unique(self._legs, "Str.__setitem__: self._legs must be unique")

  def __delitem__(self, key):
      del self._legs[key]  


  # --- Override arithmetics and comparisons ----------------------------------------------------------------

  def __eq__(self, other):
      return self._legs == other.get_str()

  def __neq__(self, other):
      return self._legs != other.get_str()

  def __add__(self, other):
      new_legs  = self.get_set() + other.get_set()
      return self.combine(other, new_legs)

  def __sub__(self, other):  
      new_legs = self.get_set() - other.get_set()
      return self.combine(other, new_legs)

  def __and__(self, other):
      new_legs  = self.get_set() & other.get_set()
      return self.combine(other, new_legs)

  def __or__(self):
      new_legs = self.get_set() | other.get_set()
      return self.combine(other, new_legs)

  def __xor__(self):
      new_legs = self.get_set() ^ other.get_set()
      return self.combine(other, new_legs)


  # --- Subsets, size, contains -----------------------------------------------------------------------------

  def find(self, leg):
      idx = self._legs.find(leg)
      return idx

  def issubset(self, other):
      self_set  = self.get_set()
      other_set = other.get_set()
      return self_set.issubset(other_set)

  def __contains__(self, other):
      return other.get_set() in self.get_set()

  def __len__(self):
      return len(self._legs) 

  # ---------------------------------------------------------------------------------------------------------





#######################################################################################################################
#######################################################################################################################
#######################################################################################################################




# --- Str auxiliary functions -------------------------------------------------------------------------------

def add_legs(legs_list):

    # Sum Str() objects
    sum_of_legs = Str("")
    for legs in legs_list:
        sum_of_legs = sum_of_legs + legs   
    return sum_of_legs 


def get_num_legs(legs_list):
    # Get num of legs in a sum of Str() objects
    return len(add_legs(legs_list))


def dict_to_list(x):

    # If x is an (A,B,C) dict, convert it to list
    if   type(x) == dict:
         return [x["A"], x["B"], x["C"]]
    else:
         return x


def subscript_to_legs(x, output_dict=True):

    # Local subscript copy
    subscript = cp.deepcopy(x)

    # Remove any spaces, add "->" if not present
    subscript = subscript.replace(' ','')
    if '->' not in subscript: subscript += '->'

    # Split up into list, using ',' as delim
    legs = subscript.replace('->', ',').split(',')
    legs = list(map(lambda ll: Str(ll), legs))

    # Convert to dict if so desired
    if  output_dict and len(legs) == 3:
        legs = {"A": legs[0], "B": legs[1], "C": legs[2]}
        
    return legs


def legs_to_subscript(legs):

    # If legs is dict, convert it to list
    legs_list = dict_to_list(legs)
    legs_list = [lg.get_str() for lg in legs_list]

    # Convert list to subscript
    subscript = ','.join(legs_list[:-1]) + '->' + legs_list[-1] 
    return subscript  


def symlegs_and_denselegs_to_subscript(symlegs, denselegs):

    # If symlegs and denselegs are dicts, convert them to lists
    symlegs_list   = dict_to_list(symlegs)
    denselegs_list = dict_to_list(denselegs)

    # Make subscript from symlegs and denselegs input
    legs_list = [symlegs_list[i].truncate() + denselegs_list[i] for i in range(len(symlegs_list))]  
    subscript = legs_to_subscript(legs_list)

    return subscript




# --- General auxiliary functions ---------------------------------------------------------------------------


def assert_equal(a, b, msg):
    assert a == b, "{}: a={}, b={}".format(msg, a, b)


def assert_included(x, vals, msg):
    assert x in vals, "{}: x={}, vals={}".format(msg, x, vals)    


def assert_unique(x, msg):
    assert_equal(len(x), len(set(x)), msg)


def isiterable(x):
    try:
        x_iterator = iter(x)
        isiter = True
    except TypeError:
        isiter = False

    return isiter

    
def multisorted(masterlist, *lists, key=None):

    # Sort input masterlist according to the key function
    if   key is not None:
         sorted_masterlist = sorted(masterlist, key=key)
    else:
         sorted_masterlist = sorted(masterlist)

    # Sort all other input lists by the masterlist
    sorted_list_func = lambda x: [x[masterlist.index(val)] for val in sorted_masterlist]
    sorted_lists     = [sorted_list_func(lst) for lst in lists]
    sorted_lists     = tuple([masterlist, *sorted_lists])

    return sorted_lists 


def del_from_dict(dct, keys):
 
    # Local copy
    x = cp.deepcopy(dct)

    # Delete dict items at given keys
    for kk in keys:
        del x[kk]
    return x


def del_from_list(lst, indices):

    # Local copy
    x = cp.deepcopy(lst)

    # Delete list items at given indices.
    # Note that you need to delete them in reverse order so that you don't
    # throw off the subsequent indices.
    for idx in sorted(indices, reverse=True):
        del x[idx]
    return x



# --- Misc auxiliary objects --------------------------------------------------------------------------------

@property
def NotImplementedField(self):
    raise NotImplementedError






































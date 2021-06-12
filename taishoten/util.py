#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools
import collections



# --- Assertions and boolean functions -------------------------------------- #

def ARE(*args):
    return all(IS(arg) for arg in args)


def ARENOT(*args):
    return all(ISNOT(arg) for arg in args)


def IS(x):
    return x is not None


def ISNOT(x):
    return x is None


def assertequal(a, b, msg):
    assert a == b, "{}: a={}, b={}".format(msg, a, b)


def assertclose(a, b, msg, rtol=2**(-16), atol=2**(-32)):
    assert np.allclose(a, b, rtol=rtol, atol=atol), msg


def assertin(x, vals, msg):
    assert x in vals, "{}: x={}, vals={}".format(msg, x, vals)    


def assertunique(x, msg):
    assertequal(len(x), len(set(x)), msg)


def isiterable(x):
    try:
        x_iterator = iter(x)
        isiter = True
    except TypeError:
        isiter = False

    return isiter


def noniterable(x):
    return not isiterable(x)



# --- Itertools wrappers ---------------------------------------------------- #

def cartesian_prod(*args):

    prod = itertools.product(*args)
    return prod



def cartesian_prod_dict(kwargs, keys):

    args = [kwargs[key] for key in keys]
    return cartesian_prod(*args)



def combinations(iterable, num_elements_per_combo):

    combos = itertools.combinations(iterable, num_elements_per_combo)
    return combos



def idx_combinations(num_elements, num_elements_per_combo):

    return combinations(range(num_elements), num_elements_per_combo)
           




# --- Str class ------------------------------------------------------------- #

class Str:

   def __init__(self, *args, **kwargs):

       # Initialize Str
       self._initialize(*args, **kwargs)


   def __call__(self, *args, **kwargs):

       # Update Str
       self._initialize(*args, **kwargs)
       return self



   # --- Initialization and new object creation ----------------------------- #

   def _initialize(self, x="", key=None, reverse=False):

       # Str input
       if  isinstance(x, type(self)):
           self._str = str(x) 
           return

       # List/tuple/ndarray input: convert to string
       if  isinstance(x, (list, tuple, np.ndarray)):
           x = to_string(x)

       # Set input: convert to sorted string
       if  isinstance(x, set):
           x = sorted_string(x, key=key, reverse=reverse)

       # Make sure our input is a string with all unique elements
       # (so that our Str can be treated as str and set simultaneously)
       assertequal(type(x), str, "Str: input type must be str")
       assertunique(x, "Str: input elements must be unique")

       # Initialize
       self._str = x



   # --- Basic operations and properties ------------------------------------ #
 
   def __str__(self):
       return self.to_str()

   def to_str(self):
       return self._str

   def to_set(self):
       return set(self._str)

   def sorted(self):
       my_set = self.to_set()
       return type(self)(my_set)

   def upper(self):
       new_str = self._str.upper() 
       return type(self)(new_str)

   def lower(self):
       new_str = self._str.lower()
       return type(self)(new_str)



   # --- Override [] access, iterator, and hashing -------------------------- #

   """
   We need to define __hash__ and __eq__ to make a class usable as a dict key:
   https://stackoverflow.com/questions/5221236/how-can-i-make-my-classes-usable-as-dict-keys

   No __setitem__ or __delitem__ as StrSet is intended to be immutable

   """

   def __getitem__(self, key):
       new_str = self._str[key]
       return type(self)(new_str)


   def __iter__(self):
       return StrIterator(self._str)


   def __hash__(self):
       return hash(self._str)



   # --- Override arithmetics and comparisons ------------------------------- #

   def __eq__(self, other):
       return self.to_str() == other.to_str()

   def __neq__(self, other):
       return self.to_str() != other.to_str()

   def __lt__(self, other):
       return self.to_str() < other.to_str()

   def __gt__(self, other):
       return self.to_str() > other.to_str()

   def __le__(self, other):
       return self.to_str() <= other.to_str()

   def __ge__(self, other):
       return self.to_str() >= other.to_str()

   def __add__(self, other):
       out = self.to_str() + other.to_str()
       out = unique_string(out) 
       return type(self)(out)

   def __sub__(self, other):  
       out = self.to_set() - other.to_set()
       return type(self)(out)

   def __and__(self, other):
       out  = self.to_set() & other.to_set()
       return type(self)(out)

   def __or__(self, other):
       out = self.to_set() | other.to_set()
       return type(self)(out)

   def __xor__(self, other):
       out = self.to_set() ^ other.to_set()
       return type(self)(out)



   # --- Subsets, size, contains --------------------------------------------- #

   def __contains__(self, other):
       return other.to_set() in self.to_set()

   def __len__(self):
       return len(self._str) 

   def find(self, x):
       idx = self._str.find(x)
       return idx

   def issubset(self, other):

       self_set  = self.to_set()
       other_set = other.to_set()

       return self_set.issubset(other_set)




class StrIterator:

   """
   Making your class iterable:
   https://stackoverflow.com/questions/21665485/how-to-make-a-custom-object-iterable

   """

   def __init__(self, x):

       self._x   = x
       self._idx = 0


   def __iter__(self):
       return self


   def __next__(self):
       
       self._idx += 1

       try:
           return Str(self._x[self._idx - 1])

       except IndexError:

           self._idx = 0
           raise StopIteration
       


def join(*args): 

    # Join Str objects
    tot = Str()
    for arg in args:
        tot += arg
    return tot




# --- Functions for iterables containing legs, nodes, maps, etc ------------- #

def dictzip(keys, vals):
    return dict(zip(keys, vals))



def dictriplet(a, b, c):
    return {"A": a, "B": b, "C": c}



def get_items(x, idx):
    return [x[i] for i in idx]



def get_legs(xs):
    return [x.legs for x in xs]



def get_from_legs(xs, legs):

    xlegs  = get_legs(xs)
    xs_dct = dictzip(xlegs, xs)

    return xs_dct[legs]



def sort_by_legs(xs, reverse=False):

    sorted_xs = sorted(xs,        key=lambda x:     x.legs)
    sorted_xs = sorted(sorted_xs, key=lambda x: len(x.legs), reverse=reverse)
    return sorted_xs



def sort(xs, reverse=False):

    sorted_xs = sorted(xs)
    sorted_xs = sorted(sorted_xs, key=len, reverse=reverse)
    return sorted_xs



def to_string(x):

    # Convert an iterable x to string
    return ''.join(str(val) for val in x)



def sorted_string(x, key=None, reverse=False):
    return ''.join(sorted(x, key=key, reverse=reverse))



def unique_string(x):
    return ''.join(collections.OrderedDict.fromkeys(x).keys())



# --- Functions for manipulating signs -------------------------------------- #

def signs_to_int(signs):

    SIGN = {'+': 1, '-': -1}
    return [SIGN[s] for s in signs]



def flip_signs(signs): 

    FLIP = {'+': '-', '-': '+', '0': '0'}
    return ''.join(FLIP[sgn] for sgn in signs)



def phased_signs(signs, phase=1):

    if  phase ==  1:
        return signs

    if  phase == -1:
        return flip_signs(signs)

    msg = "phased_signs: phase must be +1 or -1, not {}".format(phase)
    raise ValueError(msg)




# --- Functions for manipulating legs and subscripts ------------------------ #

def subscript_to_legs(subscript): 

    # Remove any spaces, add "->" if not present
    sub = subscript.replace(' ', '')
    if '->' not in sub: sub += '->'
 
    # Split up into list, using ',' as delim
    legs = sub.replace('->', ',').split(',')
    legs = [Str(x) for x in legs]
    return legs
 


def legs_to_subscript(*legs):

    # Convert list of legs to subscript 
    subscript = ','.join(str(l) for l in legs[:-1]) + '->' + str(legs[-1]) 
    return subscript  

 

def truncate(legs):

    # Function to truncate a single bunch of legs
    def trunc(x):
        return x[:-1]

    # If "legs" consists of several leg groups, 
    # truncate each one individually
    if  isinstance(legs, dict):
        return {key: trunc(legs[key]) for key in legs.keys()}

    if  isinstance(legs, (list, tuple, np.ndarray)):
        return type(legs)([trunc(ll)  for ll  in legs])

    # If "legs" consists of a single bunch
    return trunc(legs)



def cut_unsigned(x, fullsigns, unsigned='0'):

    if  isinstance(x, (list, tuple, np.ndarray)):
        return cut_unsigned_indices(x, fullsigns, unsigned=unsigned)

    if  isinstance(x, Str):
        return cut_unsigned_legs(x, fullsigns, unsigned=unsigned)

    raise ValueError("cut_unsigned: invalid x type, ".format(type(x)))



def cut_unsigned_legs(x, fullsigns, unsigned='0'):

    signed_x = ''.join(str(v) for i, v in enumerate(x) \
                                  if fullsigns[i] != unsigned)
    return Str(signed_x)



def cut_unsigned_indices(idx, fullsigns, unsigned='0'):

    signed_idx = []
    num_cut    = 0

    for i in idx:

        if  fullsigns[i] == unsigned:
            num_cut += 1
            continue
  
        signed_idx.append(i - num_cut)

    return signed_idx

    

def get_num_legs(*args):

    # Get num of (unique) legs in a sum of Str objects
    return len(join(*args))



def make_legs(num_legs): 
    ALPHABET = 'abcdefghijklmnoprstuvwxyz'
    legs = Str(ALPHABET[:num_legs])
    return legs



def make_symlegs(legs, fullsigns):

    # Create symlegs from denselegs
    symlegs = legs.upper() 
    symlegs = cut_unsigned(symlegs, fullsigns)
    return symlegs




# --- Generators for leg indices and data ----------------------------------- #

def zip_compress(a, b, selectors):

    legsA = a[0]
    legsB = b[0]

    itemsA = dictzip(*a)
    itemsB = dictzip(*b)

    for leg in selectors:

        if   leg in itemsA:
             yield itemsA[leg]
        else:
             yield itemsB[leg]



def zip_shared(a, b):

    legsA    = a[0]
    legsB    = b[0]
    sharedAB = legsA & legsB

    itemsA = dictzip(*a)
    itemsB = dictzip(*b)

    for leg in sharedAB:            
        yield itemsA[leg], itemsB[leg]



def get_shared_indices(legsA, legsB):

    num_legsA = len(legsA)
    num_legsB = len(legsB)

    idxA = (legsA, range(num_legsA))
    idxB = (legsB, range(num_legsB))    

    shared_idx_AB              = list(zip_shared(idxA, idxB))
    shared_idx_A, shared_idx_B = zip(*shared_idx_AB)

    return shared_idx_A, shared_idx_B





































































































































































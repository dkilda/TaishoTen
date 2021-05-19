#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools



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


def assertclose(a, b, msg, rtol=SYMTOL, atol=SYMTOL**2):
    assert np.allclose(a, b, rtol=rtol, atol=atol), msg


def assertin(x, vals, msg):
    assert x in vals, "{}: x={}, vals={}".format(msg, x, vals)    


def assertunique(x, msg):
    assert_equal(len(x), len(set(x)), msg)


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
           




# --- Operations on iterable objects ---------------------------------------- #

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
    # Note that you need to delete them in reverse order
    # so that you don't throw off the subsequent indices.
    for idx in sorted(indices, reverse=True):
        del x[idx]
    return x



def to_string(x):

    # Convert an iterable x to string
    return ''.join(str(val) for val in x)







# --- Miscellaneous functions ----------------------------------------------- #

@property
def NotImplementedField(self):
    raise NotImplementedError




# --- StrSet class ---------------------------------------------------------- #


class StrSet:

   def __init__(self, *args, **kwargs):

       # Initialize StrSet
       self._initialize(*args, **kwargs)


   def __call__(self, *args, **kwargs):

       # Update StrSet
       self._initialize(*args, **kwargs)
       return self
       

   # --- Initialization and new object creation ----------------------------- #

   def _initialize(self, x="", order=None)

       # If input is StrSet --> convert to string
       if  isinstance(x, type(self)):
           x = x.to_str() 

       # If input is a string --> initialize and return
       if  isinstance(x, str):
           self._initialize_from_string(x)
           return

       # If input is a list/tuple/ndarray --> convert to string
       if  isinstance(x, (list, tuple, np.ndarray)):
           x = to_string(x)

       # If input is a set --> convert to string
       if  isinstance(x, set):

           # Default order
           if  ISNOT(order):
               order = {val: i for i, val in enumerate(x)}
           
           # Convert set to string, using the order provided
           sorted_x = sorted(x, key=lambda k: order[k])
           x        = to_string(sorted_x)

       # By this point, input should have been converted to string
       # (or is completely invalid)
       self._initialize_from_string(x)



   def _initialize_from_string(self, x):

       # Make sure our input is str
       assertequal(type(x), str, "StrSet: input type must be str")

       # Make sure all input elements are unique, 
       # so that our StrSet can be treated as str and set simultaneously
       assertunique(x, "StrSet: input elements must be unique")

       # Set attributes
       self._str = x



   # --- Basic operations and properties ------------------------------------ #
 
   def __str__(self):
       return self.to_str()

   def to_str(self):
       return self._str

   def to_set(self):
       return set(self._str)

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
       return StrSetIterator(self._str)


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
       combo = self.to_set() + other.to_set()
       return combine_strsets(self, other, combo)

   def __sub__(self, other):  
       combo = self.to_set() - other.to_set()
       return combine_strsets(self, other, combo)

   def __and__(self, other):
       combo  = self.to_set() & other.to_set()
       return combine_strsets(self, other, combo)

   def __or__(self, other):
       combo = self.to_set() | other.to_set()
       return combine_strsets(self, other, combo)

   def __xor__(self, other):
       combo = self.to_set() ^ other.to_set()
       return combine_strsets(self, other, combo)



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





class StrSetIterator:

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
           return StrSet(self._x[self._idx - 1])

       except IndexError:
           self._idx = 0
           raise StopIteration
       





# --- Functions operating on StrSet ----------------------------------------- #


def combine_strsets(x, y, combo_xy):

    """
    combo_xy: can be any iterable (set, str, list, etc) 
              obtained after some binary operation 
              that combines elements from x and y.
    """

    # Since combo_xy may be unordered or ordered incorrectly,
    # we order its elements according to input "string"
    def ordered_str(string):
        return ''.join(s for s in string if s in combo_xy)

    # Take combo_xy elements \in x, order them according to x.
    # Take combo_xy elements \in y, order them according to y.
    # Add both strings together so that 
    # new_str = "combo_xy elements \in x, combo_xy elements \in y"
    new_str = ordered_str(str(x)) + ordered_str(str(y))

    # Create a new StrSet
    return StrSet(new_str)



def sum_strsets(*args):

    # Sum StrSet objects
    tot = StrSet()
    for arg in args:
        tot += arg
    return tot




# --- Functions for manipulating legs and subscripts ------------------------ #


def subscript_to_legs(subscript): 

    # Remove any spaces, add "->" if not present
    sub = subscript.replace(' ', '')
    if '->' not in sub: sub += '->'
 
    # Split up into list, using ',' as delim
    legs = sub.replace('->', ',').split(',')
    legs = [StrSet(x) for x in legs]
    return legs
 


def legs_to_subscript(*legs):

    # Convert list of legs to subscript 
    subscript = ','.join(legs[:-1].to_str()) + '->' + legs[-1].to_str() 
    return subscript  

    

def sym_dense_legs_to_subscript(symlegs, denselegs, truncated=False): 

    # Truncate input symlegs
    if not truncated:
       symlegs = truncate(symlegs)

    # Make subscript from symlegs and denselegs input
    legs      = [ss + dd for ss, dd in zip(symlegs, denselegs)]
    subscript = legs_to_subscript(*legs)    
    return subscript



def truncate(legs):

    # Function to truncate a single bunch of legs
    def trunc(x):
        return x[:-1]

    # If "legs" consists of several leg groups, 
    # truncate each one individually
    if  isinstance(legs, (list, tuple, np.ndarray)):
        trunc_legs = type(legs)([trunc(ll) for ll in legs])
        return trunc_legs

    # If "legs" consists of a single bunch
    return trunc(legs)



def cut_unsigned(legs, fullsigns, unsigned='0'):

    # Cut unsigned legs 
    new_legs = ''.join(v for i, v in enumerate(legs) \
                                  if fullsigns[i] != unsigned)
    return StrSet(new_legs)



def get_num_legs(*args):

    # Get num of (unique) legs in a sum of StrSet objects
    return len(sum_strsets(*args))



def make_symlegs(denselegs, fullsigns):

    # Create symlegs from denselegs
    symlegs = denselegs.upper() 
    symlegs = cut_unsigned(symlegs, fullsigns)
    return symlegs








# --- Generators for leg indices and data ----------------------------------- #


def gen_data(src, dest, dest_data): 

    # Take legs from "src", 
    # get their indices in "dest" and their data in "dest_data"
    for idx in gen_idx(src, dest):
        yield dest_data[idx]

 

def gen_idx(src, dest): 
 
    # Take legs from "src", get their indices in "dest"
    for s in src:
        if s in dest:
           idx = dest.index(s)
           yield idx



def gen_binary_data(conjunction):

    """
    Extension of gen_data to "dest"/"dest_data" with two 
    elements, i \in (0,1). The "dest" data from different
    elements can be combined in two ways: AND or OR. 
 
    """
    
    def _gen_binary_data(src, dest, dest_data):

        # Retrieve data of leg "s" in dest component "i" 
        def data(s, i):
            idx = dest[i].index(s) 
            dat = dest_data[i][idx]
            return dat


        if   conjunction == "AND":

             # Yield data from both elements i=0,1 (AND)
             for s in src:
                 yield data(s,0), data(s,1) 

        elif conjunction == "OR":

             # Yield data from one of the elements (OR)
             for s in src:
                 if   s in dest[0]:
                      yield data(s,0)
                 else:
                      yield data(s,1)

        else:
             msg = "gen_binary_data: invalid conjunction {}"
             msg = msg.format(conjunction)
             raise ValueError(msg)

    return _gen_binary_data







































































































































































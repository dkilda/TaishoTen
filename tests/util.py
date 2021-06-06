#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'..')

import os
import numpy as np
import numpy.testing as nptest
import unittest

import taishoten as tn
from taishoten import Str

SAVE_DATA_DIR = "./data/"
LOAD_DATA_DIR = "./tests/data/"

CLOSE_RTOL = 2**(-16)
CLOSE_ATOL = 2**(-32)


# --- Auxiliary functions --------------------------------------------------- #


def isiterable(x):
    try:
        x_iterator = iter(x)
        isiter = True
    except TypeError:
        isiter = False

    return isiter



def noniterable(x):
    return not isiterable(x)



def save_array(filename, array):

    filename = "test_{}.npy".format(filename)
    path = os.path.join(SAVE_DATA_DIR, filename)

    return np.save(path, array, allow_pickle=True)



def load_array(filename):

    filename = "test_{}.npy".format(filename)
    path = os.path.join(LOAD_DATA_DIR, filename)

    return np.load(path, allow_pickle=True)



def must_fail(fun, error=AssertionError):

    def fwrap(*args, **kwargs):

        try:
           fun(*args, **kwargs)
           failed = False          
        except error:
           failed = True

        assert failed

    return fwrap




# --- TaishoTen test case class --------------------------------------------- #

class TaishoTenTestCase(unittest.TestCase):


   # --- Basic assertions --------------------------------------------------- #

   def assertEqual(self, array, ans):
       nptest.assert_array_equal(array, ans)

   def assertClose(self, array, ans):
       nptest.assert_allclose(array, ans, rtol=CLOSE_RTOL, atol=CLOSE_ATOL)

   def assertEqualArray(self, array, ans):
       self.assertEqual(array, ans)
       self.assertEqual(array.shape, ans.shape)

   def assertCloseArray(self, array, ans):
       self.assertClose(array, ans)
       self.assertEqual(array.shape, ans.shape)



   # --- Lists and dicts ---------------------------------------------------- # 

   def assertList(self, lst, ans, fun=None):

       if fun is None:
          def fun(x,y): return x == y

       assert len(lst) == len(ans)
       for l, a  in zip(lst, ans):
           fun(l, a)


   def assertDict(self, dct, ans, fun=None):

       if fun is None:
          def fun(x,y): return x == y 

       keys = list(dct.keys())

       assert len(dct) == len(ans)
       assertList(keys, list(ans.keys()))

       for key in keys:
           fun(dct[key], ans[key])


   def assertEqualList(self, lst, ans):
       self.assertList(lst, ans, fun=self.assertEqual)

   def assertCloseList(self, lst, ans):
       self.assertList(lst, ans, fun=self.assertClose)

   def assertEqualArrayList(self, lst, ans):
       self.assertList(lst, ans, fun=self.assertArrayEqual)

   def assertCloseArrayList(self, lst, ans):
       self.assertList(lst, ans, fun=self.assertArrayClose)



   # --- Maps --------------------------------------------------------------- #

   def assertMap(self, x, legs, shape, dtype=np.ndarray):

       assert type(x)       == tn.Map
       assert type(x.array) == dtype
       assert x.legs        == legs
       assert x.shape       == shape
       assert x.ndim        == len(shape)


   def assertMapVsArray(self, x, legs, array):

       self.assertMap(x, legs, array.shape, type(array))
       self.assertCloseArray(x.array, array)


   def assertEqualMap(self, x, ans):

       self.assertMap(x, ans.legs, ans.shape, type(ans.array))
       self.assertCloseArray(x.array, ans.array)



   # --- Symmetries --------------------------------------------------------- #

   def assertSymmetry(self, x, fullsigns, symlabels, \
                            qtot=0, mod=None, signs=None, ndim=1):

       if  signs is None:
           signs = fullsigns  

       assert type(x) is {1: tn.Symmetry1D, 3: tn.Symmetry3D}[ndim]

       assert x.fullsigns == fullsigns
       assert x.signs     == signs

       assert x.qtot  == qtot
       assert x.mod   == mod
       assert x.ndim  == ndim

       assert x.num_legs    == len(fullsigns)
       assert x.num_symlegs == len(signs)

       assertList(x.symlabels, symlabels, lambda l, a: type(l) == type(a))
       self.assertCloseArrayList(x.symlabels, symlabels)


   def assertEqualSymmetry(self, x, ans):

       if  ans is None:
           assert x is None
           return 

       self.assertSymmetry(x, ans.fullsigns, ans.symlabels, \
                              ans.qtot, ans.mod, ans.signs, ans.ndim)



   # --- Symmetry contraction ----------------------------------------------- #

   def assertSymmetryContraction(self, x, sym, legs, symlegs, phase):

       assert type(x) == tn.SymmetryContraction
       assert x.type  == type(sym["A"])
       assert x.type  == type(sym["B"])
       assert x.ndim  == sym["A"].ndim
       assert x.ndim  == sym["B"].ndim
       assert x.phase == phase

       assertDict(x.sym(),     sym,     fun=self.assertEqualSymmetry)
       assertDict(x.legs(),    legs,    fun=self.assertEqualStr)
       assertDict(x.symlegs(), symlegs, fun=self.assertEqualStr)

       for key in ("A", "B", "C"):

           self.assertEqualSymmetry(x.sym(key), sym[key])
           self.assertEqualStr(x.symlegs(key), symlegs[key])
           self.assertEqualStr(x.legs(key), legs[key])

       

   def assertEqualSymmetryContraction(self, x, ans):

       self.assertSymmetryContraction(x, ans.sym(),     ans.legs(), \
                                         ans.symlegs(), ans.phase)   



   # --- Tensor ------------------------------------------------------------- #

   def assertTensor(self, x, shape, dtype=np.ndarray, sym=None):

       assert type(x)       == tn.Tensor
       assert type(x.array) == dtype
       assert x.shape       == shape
       assert x.ndim        == len(shape)
       
       if   sym is None:

            assert x.sym is None
            assert x.sym_shape is None
            assert x.dense_shape == shape

       else: 
            self.assertEqualSymmetry(x.sym, sym)
            assert x.sym_shape   == shape[: sym.num_symlegs]
            assert x.dense_shape == shape[sym.num_symlegs :]


   def assertTensorVsArray(self, x, array, sym=None):

       self.assertTensor(x, array.shape, dtype=type(array), sym=sym)
       self.assertCloseArray(x.array, array)


   def assertEqualTensor(self, x, ans): 

       self.assertTensor(x, ans.shape, dtype=type(ans.array), sym=ans.sym)
       self.assertCloseArray(x.array, ans.array)



   # --- Transformations ---------------------------------------------------- #
       
   def assertNode(self, x, legs_, map_=None, reversed_=False):

       assert type(x)    == tn.transformations.Node
       assert x.legs     == legs_
       assert x.reversed == reversed_

       if   map_ is None:

            assert x.end_legs   == legs_
            assert x.start_legs == legs_

            assert x.map       is None
            assert x.map_legs  is None
            assert x.map_array is None
            assert x.previous  is None

            assert x.layer == 0

       else:
            assert x.previous is not None
            assert node.start_legs == x.previous.end_legs

            self.assertEqualMap(x.map,     map_)
            self.assertCloseArray(x.array, map_.array)
            assert x.map_legs == map_.legs

            if    reversed_:
                  assert x.start_legs == legs_
                  assert x.end_legs   == x.previous.legs
            else:
                  assert x.start_legs == x.previous.legs
                  assert x.end_legs   == legs_

            assert x.layer == x.previous.layer + 1



   def assertEqualNode(self, x, ans):
       self.assertNode(x, ans.legs, ans.map, ans.reversed)



   def assertGraph(self, x, nodes, legs, num_out_symlegs):

       assert type(x) == tn.transformations.TransformGraph

       assertList(x.get_nodes(), nodes, fun=self.assertEqualNode)
       assertList(x.nodes,       nodes, fun=self.assertEqualNode)
       assertList(x.legs,        legs,  fun=self.assertEqual)

       assert x.num_nodes       == len(nodes)
       assert x.num_out_symlegs == num_out_symlegs

     

   def assertEqualGraph(self, x, ans):

       assertList(x.maps, ans.maps, fun=self.assertEqualMap)
       assert x.symlegs == ans.symlegs

       self.assertGraph(x, ans.nodes, ans.legs, ans.num_out_symlegs)



   def assertPathlet(self, x, maps, start_symlegs, end_symlegs, reverse=False):

       # Check pathlet is the correct data structure
       assert type(x) in (list, tuple)
       if  len(x) == 0:
           return

       # Check start and end pts of the pathlet
       if   reverse:
            assert x[-1].end_legs  == start_symlegs[:-1]
            assert x[0].start_legs == end_symlegs[:-1]
       else:
            assert x[-1].end_legs  == end_symlegs[:-1]
            assert x[0].start_legs == start_symlegs[:-1]

       # Check every node
       for i, node in enumerate(x):

           # Type and basic properties
           assert type(node) is tn.Node
           assert node.reverse == reverse
           assert node.layer   == i + 1

           # Map
           node_map = node.map
           ans_map  = tn.get_from_legs(maps, node.map_legs)
           self.assertEqualMap(node_map, ans_map)

           # End legs of current node and start legs of next node must match
           if  i < len(path) - 1:
               assert path[i].end_legs == path[i+1].start_legs

           # End legs of the node must be valid and
           # match the output legs given by map transformation
           start_legs = node.start_legs
           end_legs   = node.end_legs
           map_legs   = node.map_legs
        
           ALL    = start_legs | map_legs
           SHARED = start_legs & map_legs
           EXTRA  = ALL - SHARED

           HADAMARD = SHARED & end_legs
           DOT      = SHARED - HADAMARD
           OUT      = EXTRA | HADAMARD

           assert     end_legs == OUT
           assert     end_legs.issubset(ALL)
           assert not DOT.issubset(end_legs)

       

   def assertEqualPathlet(self, x, ans):

       assertList(x, ans, fun=self.assertEqualNode)



   def assertPath(self, x, maps, start_symlegs, end_symlegs):

       start = start_symlegs
       end   = end_symlegs

       self.assertPathlet(x["A"], maps, start["A"], end["A"])
       self.assertPathlet(x["B"], maps, start["B"], end["B"])
       self.assertPathlet(x["C"], maps, start["C"], end["C"], reverse=True)

       legs = {"A": x["A"][-1].legs, \  
               "B": x["B"][-1].legs, \ 
               "C": x["C"][0].legs   }

       num_ind_symlegs = tn.get_num_ind_symlegs(start_symlegs)
       assert tn.good_to_contract(legs, num_ind_symlegs)   



   def assertEqualPath(self, x, ans): 

       assertDict(x, ans, fun=self.assertEqualPathlet)



   # --- Str object --------------------------------------------------------- #

   def assertStr(self, x, string):

       # Check size and str/set conversions
       assert len(x)     == len(string)
       assert str(x)     == string
       assert set(x)     == set(string)
       assert x.to_str() == string
       assert x.to_set() == set(string)

       # Check sorting
       assert x.sorted() == ''.join(sorted(string))

       # Check "in" and "issubset"
       assert all(s in x for s in string) 
       assert x.issubset(Str(string))

       # Check "__getitem__", "find", "issubset" for
       # individual elements of Str
       for i, s in enumerate(string):
    
           s1 = Str(s)

           assert x[i] == s1
           assert x.find(s1) == i
           assert s1.issubset(x)

       # Check uppercase/lowercase conversions
       assert x.upper() == string.upper()
       assert x.lower() == string.lower()


   def assertEqualStr(self, x, ans):

       if  ans is None:
           assert x is None
           return

       self.assertStr(x, str(ans))
       


   # --- Symeinsum ---------------------------------------------------------- #

   def assertSymeinsum(self, A, B, sub, sub1, sub2):
       
       # Get output symmetry and map
       legs = tn.subscript_to_legs(subscript)
       legs = tn.dictriplet(*legs)

       symcon = tn.compute_symmetry_contraction(A.sym, B.sym, legs)
       symC   = symcon.sym("C")
       mapC   = tn.Map.compute(symC, symcon.symlegs("C"))

       # Tensor C from tn.symeinsum
       out = tn.symeinsum(sub, A, B)

       # Array C from np.einsum
       arrayA = A.get_full_array()
       arrayB = B.get_full_array()

       arrayC = np.einsum(sub1, arrayA, arrayB)
       arrayC = np.einsum(sub2, arrayC, mapC.array)

       # Test
       self.assertTensorVsArray(out, arrayC, symC)















































































































































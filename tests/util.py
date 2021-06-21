#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'..')

import numpy as np
import numpy.testing as nptest
import pytest

import taishoten as tn
from taishoten import Str



# --- Basic assertions ------------------------------------------------------ #

def assert_array_equal(x, ans):
    nptest.assert_array_equal(x, ans)


def assert_array_close(x, ans, rtol=2**(-16), atol=2**(-32)):
    nptest.assert_allclose(x, ans, rtol=rtol, atol=atol)


def assert_array_list_equal(x, ans):
    assert_list(x, ans, fun=assert_array_equal)


def assert_array_list_close(x, ans):
    assert_list(x, ans, fun=assert_array_close)


def assert_list(x, ans, fun=None):

    if fun is None:
       def fun(v, u): assert v == u

    assert len(x) == len(ans)
    for vx, va in zip(x, ans):
        fun(vx, va)


def assert_dict(x, ans, fun=None):

    if fun is None:
       def fun(v, u): assert v == u

    keys     = sorted(x.keys())
    ans_keys = sorted(ans.keys())

    assert len(x) == len(ans)
    assert_list(keys, ans_keys)

    for key in keys:
        fun(x[key], ans[key])




# --- Str object ------------------------------------------------------------ #

def assert_Str(x, string):

    # Check size and str/set conversions
    assert len(x)     == len(string)
    assert str(x)     == string

    assert x.to_set() == set(string)
    assert x.to_str() == string
    assert x.to_set() == set(string)

    # Check sorting
    assert str(x.sorted()) == ''.join(sorted(string))

    # Check "in" and "issubset"
    assert all(s in x for s in Str(string)) 
    assert x.issubset(Str(string))

    # Check "__getitem__", "find", "issubset" for
    # individual elements of Str
    for i, s in enumerate(string):
    
        s1 = Str(s)

        assert x[i] == s1
        assert x.find(s1) == i
        assert s1.issubset(x)

    # Check uppercase/lowercase conversions
    assert str(x.upper()) == string.upper()
    assert str(x.lower()) == string.lower()



def assert_Str_equal(x, ans):

    if  ans is None:
        assert x is None
        return

    assert_Str(x, str(ans))




# --- Maps ------------------------------------------------------------------ #

def assert_map(x, legs, shape, dtype=np.ndarray):

    assert type(x)       == tn.Map
    assert type(x.array) == dtype
    assert x.legs        == legs
    assert x.shape       == shape
    assert x.ndim        == len(shape)



def assert_map_vs_array(x, legs, array):

    assert_map(x, legs, array.shape, type(array))
    assert_array_close(x.array, array)



def assert_map_equal(x, ans):

    assert_map(x, ans.legs, ans.shape, type(ans.array))
    assert_array_close(x.array, ans.array)




# --- Symmetries ------------------------------------------------------------ #

def assert_symmetry(x, fullsigns, symlabels, \
                       qtot=0, mod=None, signs=None, ndim=1):

    if  signs is None:
        signs = fullsigns  

    assert type(x) is {1: tn.Symmetry1D, 3: tn.Symmetry3D}[ndim]

    assert x.ndim == ndim

    assert x.fullsigns == fullsigns
    assert x.signs     == signs

    assert x.num_legs    == len(fullsigns)
    assert x.num_symlegs == len(signs)

    assert_array_close(x.qtot, qtot)

    if   mod is None:
         assert x.mod is None
    else:
         assert_array_close(x.mod, mod)

    symlabel_types = [np.ndarray]*len(symlabels)
    assert_list(x.symlabels, symlabel_types, lambda u,v: type(u) == v)

    assert_array_list_close(x.symlabels, symlabels)



def assert_symmetry_equal(x, ans):

    if  ans is None:
        assert x is None
        return 

    assert_symmetry(x, ans.fullsigns, ans.symlabels, \
                       ans.qtot, ans.mod, ans.signs, ans.ndim)




# --- Symmetry contraction -------------------------------------------------- #

def assert_symmetry_contraction(x, sym, legs, symlegs, phase):

    assert type(x) == tn.SymmetryContraction
    assert x.type  == type(sym["A"])
    assert x.type  == type(sym["B"])
    assert x.ndim  == sym["A"].ndim
    assert x.ndim  == sym["B"].ndim
    assert x.phase == phase

    assert_dict(x.sym(),     sym,     fun=assert_symmetry_equal)
    assert_dict(x.legs(),    legs,    fun=assert_Str_equal)
    assert_dict(x.symlegs(), symlegs, fun=assert_Str_equal)

    for key in ("A", "B", "C"):

        assert_symmetry_equal(x.sym(key), sym[key])
        assert_Str_equal(x.symlegs(key), symlegs[key])
        assert_Str_equal(x.legs(key), legs[key])

       

def assert_symmmetry_contraction_equal(x, ans):

    assert_symmetry_contraction(x, ans.sym(), ans.legs(), \
                                   ans.symlegs(), ans.phase)   




# --- Tensor ---------------------------------------------------------------- #

def assert_tensor(x, shape, dtype=np.ndarray, sym=None):

    assert type(x)       == tn.Tensor
    assert type(x.array) == dtype
    assert x.shape       == shape

    if   sym is None:

         assert x.sym is None
         assert x.sym_shape is None
         assert x.dense_shape == shape
         assert x.ndim        == len(shape)

    else: 
         assert_symmetry_equal(x.sym, sym)
         assert x.sym_shape   == shape[: sym.num_symlegs - 1]
         assert x.dense_shape == shape[sym.num_symlegs - 1 :]
         assert x.ndim        == sym.num_legs



def assert_tensor_vs_array(x, array, sym=None):

    assert_tensor(x, array.shape, dtype=type(array), sym=sym)
    assert_array_close(x.array, array)



def assert_tensor_equal(x, ans): 

    assert_tensor(x, ans.shape, dtype=type(ans.array), sym=ans.sym)
    assert_array_close(x.array, ans.array)



# --- Transformations ------------------------------------------------------- #

def assert_node(x, legs_, map_=None, reversed_=False):

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
         
         assert_map_equal(x.map, map_)
         assert_array_close(x.map_array, map_.array)
         assert x.map_legs == map_.legs

         if   reversed_:
              assert x.start_legs == legs_
              assert x.end_legs   == x.previous.legs
         else:
              assert x.end_legs   == legs_
              assert x.start_legs == x.previous.legs
              assert x.start_legs == x.previous.end_legs

         assert x.layer == x.previous.layer + 1



def assert_node_equal(x, ans):
    assert_node(x, ans.legs, ans.map, ans.reversed)



def assert_graph(x, nodes, legs, num_out_symlegs):

    assert type(x) == tn.transformations.TransformGraph

    assert_list(x.get_nodes(), nodes, fun=assert_node_equal)
    assert_list(x.nodes,       nodes, fun=assert_node_equal)
    assert_list(x.legs,        legs)

    assert x.num_nodes       == len(nodes)
    assert x.num_out_symlegs == num_out_symlegs

     

def assert_graph_equal(x, ans):

    assert x.symlegs == ans.symlegs
    assert_list(x.maps, ans.maps, fun=assert_map_equal)

    assert_graph(x, ans.nodes, ans.legs, ans.num_out_symlegs)



def assert_pathlet(x, maps, start_symlegs, end_symlegs, reversed_=False):

    # Check pathlet is the correct data structure
    assert type(x) in (list, tuple)
    if  len(x) == 0:
        return

    # Check start and end pts of the pathlet
    if   reversed_:
         assert x[-1].end_legs  == start_symlegs
         assert x[0].start_legs == end_symlegs

    else:
         assert x[-1].end_legs  == end_symlegs
         assert x[0].start_legs == start_symlegs

    # Check every node
    for i, node in enumerate(x):

        # Type and basic properties
        assert type(node) is tn.transformations.Node
        assert node.reversed == reversed_

        layer = len(x) - i if reversed_ else i + 1
        assert node.layer == layer

        # Map
        node_map = node.map
        ans_map  = tn.util.get_from_legs(maps, node.map_legs)
        assert_map_equal(node_map, ans_map)

        # End legs of current node and start legs of next node must match
        if  i < len(x) - 1:
            assert x[i].end_legs == x[i+1].start_legs

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

       

def assert_pathlet_equal(x, ans):
    assert_list(x, ans, fun=assert_node_equal)



def assert_path(x, maps, start_symlegs, end_symlegs, num_ind_symlegs):

    start = start_symlegs
    end   = end_symlegs

    assert_pathlet(x["A"], maps, start["A"], end["A"])
    assert_pathlet(x["B"], maps, start["B"], end["B"])
    assert_pathlet(x["C"], maps, start["C"], end["C"], reversed_=True)

    legs = {"A": x["A"][-1].legs, \
            "B": x["B"][-1].legs, \
            "C": x["C"][0].legs   }

    assert tn.transformations.good_to_contract(legs, num_ind_symlegs)   



def assert_path_equal(x, ans): 
    assert_dict(x, ans, fun=assert_pathlet_equal)




# --- Symeinsum ------------------------------------------------------------- #

def assert_symeinsum(A, B, sub, sub1, sub2):
       
    # Get output symmetry and map
    legs = tn.subscript_to_legs(sub)
    legs = tn.dictriplet(*legs)

    symcon = tn.compute_symmetry_contraction(A.sym, B.sym, legs)
    symC   = symcon.sym("C")
    mapC   = tn.Map.compute(symC, symcon.symlegs("C"))

    # Tensor C from tn.symeinsum
    out = tn.symeinsum(sub, A, B)

    # Array C from np.einsum
    arrayA = A.get_full_array()
    arrayB = B.get_full_array()

    print("\nASSERT SYMEINSUM: ", sub1, arrayA.shape, arrayB.shape)

    arrayC = np.einsum(sub1, arrayA, arrayB)
    arrayC = np.einsum(sub2, arrayC, mapC.array)

    # Test
    print("\nASSERT: ", out.shape, arrayC.shape)

    assert_tensor_vs_array(out, arrayC, symC)








































































































































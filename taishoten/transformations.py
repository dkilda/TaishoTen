#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools

import taishoten.util as util

from taishoten.util import Str, dictriplet
from taishoten.util import assertequal
from taishoten.util import IS, ISNOT, ARE, ARENOT



# --- Finding transformation path ------------------------------------------- #

def find_transform_path(maps, symlegs): 

    # Get symleg dims and the number of independent symlegs
    symleg_dims     = get_symleg_dims(maps)
    num_ind_symlegs = get_num_ind_symlegs(symlegs)

    # Build complete transformation node lists for A, B, C tensors
    graph_A = build_transform_graph(symlegs["A"], maps) 
    graph_B = build_transform_graph(symlegs["B"], maps)
    graph_C = build_transform_graph(symlegs["C"], maps)
    
    # Get all combinations of nodes a, b, c 
    # (corresponding to tensors A, B, C). Try them as final nodes.
    final_nodes = []
    costs       = []

    for a, b, c in util.cartesian_prod(graph_A.nodes, \
                                       graph_B.nodes, \
                                       graph_C.nodes):

        # Candidate combination of the final nodes (a,b,c)
        node_legs = dictriplet(a.legs, b.legs, c.legs)

        # Check if (a,b,c) combo is valid (symlegs can be directly contracted)
        if  good_to_contract(node_legs, num_ind_symlegs): 
            
            # Record this (a,b,c) combo.
            # Get the cost of contracting the final a, b symlegs.
            cost = get_cost(a, b, symleg_dims)
            costs.append(cost)
            final_nodes.append((a, b, c))

    # Find the lowest cost combination of final nodes (a,b,c)
    idx     = costs.index(min(costs))
    a, b, c = final_nodes[idx] 

    # Compute transformation pathlets of A,B,C tensors: 
    # from the initial symlegs combo to the final combo (a,b,c) 
    pathlet_A = find_transform_pathlet(a)
    pathlet_B = find_transform_pathlet(b)         
    pathlet_C = find_transform_pathlet(c, reverse=True)    

    path       = dictriplet(pathlet_A, pathlet_B, pathlet_C)
    final_legs = dictriplet(a.legs, b.legs, c.legs)

    return path, final_legs 



# --- Auxiliary functions for find_transform_path --------------------------- #

def get_cost(a, b, leg_dims):

    # Get cost of contracting the final a, b legs
    legs = a.end_legs | b.end_legs
    cost = np.prod([leg_dims[leg] for leg in legs])
 
    return cost



def good_to_contract(symlegs, num_ind_symlegs=None): 

    # If number of independent symlegs is not given, treat symlegs as
    # untruncated: find num_ind_symlegs internally, and then truncate symlegs.
    truncated = IS(num_ind_symlegs)
    if  not truncated:  
        num_ind_symlegs = get_num_ind_symlegs(symlegs) 
        symlegs         = util.truncate(symlegs) 

    # Input and output symlegs
    IN  = symlegs["A"] + symlegs["B"]
    OUT = symlegs["C"]

    # Two conditions for direct contractability, (1) on input, (2) on output:
    #
    # (1) num of input legs = num of independent legs: 
    #     input does not contain any dependent legs, len(IN) > num_ind_symlegs
    #     and   does not omit  any independent legs, len(IN) < num_ind_symlegs
    # 
    # (2) output legs \in input legs: 
    #     output does not have any new legs not present in the input

    is_contractable = (len(IN) == num_ind_symlegs) and OUT.issubset(IN) 
    return is_contractable                              
                               


def get_num_ind_symlegs(symlegs): 

    # Sort symleg groups by length: must iterate groups with unseen legs 
    # from-shortest-to-longest to count the number of dependent legs correctly
    sorted_symlegs = util.sort(symlegs.values())

    # Count the dependent legs
    num_dep_symlegs = 0
    visited_symlegs = Str()

    for symlg in sorted_symlegs:

        # There's one dependent leg in each (group containing unvisited legs)
        if  not symlg.issubset(visited_symlegs):
            visited_symlegs = visited_symlegs | symlg
            num_dep_symlegs += 1
 
    # Get the number of independent symlegs
    num_symlegs     = util.get_num_legs(*sorted_symlegs)
    num_ind_symlegs = num_symlegs - num_dep_symlegs
    return num_ind_symlegs
    


def get_symleg_dims(maps):  

    # Get dimensions of all symlegs from map shapes. NB we cannot use
    # the same approach as for dense legs, because "Q" auxiliary legs 
    # are only present in map shapes but not tensor shapes 

    # Generator that gives dims of all map legs
    def gen_mapleg_dims(maps):
        for mp in maps:
          for leg, dim in zip(mp.legs, mp.shape):
              yield leg, dim

    # Get symlegs dims from map leg dims
    symleg_dims = {leg: dim for leg, dim in gen_mapleg_dims(maps)}
    return symleg_dims




# --- Find transform pathlet ------------------------------------------------ #

def find_transform_pathlet(final, reverse=False):

    # Compute a path from initial to final node 
    # in the transformation graph. As long as we take an existing
    # final node in the graph, the path will lead us to the initial node.
    path = []
    current = final

    # Starting from the final node, travel down the graph 
    # until we reach the initial node (with map = None).
    # If we want a reversed path from final to initial node instead, 
    # make a path of reversed nodes (with start and end legs swapped).
    while current.map:
             
       if   reverse:
            path.append(current.reversed_copy())
       else:
            path.append(current.copy())

       current = current.previous

    # Reverse the path to get the initial-to-final order
    # (unless we mean to keep it reversed, i.e. reverse=True).
    if not reverse:
       path = list(reversed(path))

    return path





# --- Transformation graph -------------------------------------------------- #


def build_transform_graph(symlegs, maps):

    # Initializes and builds transformation graph in one step
    graph = TransformGraph(symlegs, maps)
    graph.build()
    return graph
    


class TransformGraph:

   def __init__(self, symlegs, maps):

       # Truncate symlegs input
       symlegs = util.truncate(symlegs)

       # Sort the maps by the length of their symlegs 
       maps = util.sort_by_legs(maps)

       # Initialize
       self._symlegs = symlegs
       self._maps    = maps
       self._nodes   = []

       maplegs       = util.get_legs(self.maps)
       self._maplegs = util.sort([ll.sorted() for ll in maplegs])
       self._legs    = []

       self._num_out_symlegs = self.get_num_out_symlegs()



   def get_num_out_symlegs(self): 

       # Get number output symlegs 
       # (after transforming input "symlegs" using "maps")

       # For every map whose legs \in input "symlegs", one leg in "symlegs"
       # is a dependent leg (i.e. if map legs are subset of input symlegs, 
       # we have an extra conservation law with map as a node, which implies
       # that one of input symlegs is constrained by this law and is dependent)
       #
       # Otherwise (without the constraints above), we will have 
       # "num independent output symlegs" = "num independent input symlegs"
       # because each map can only sum out one leg or replace one leg by another
       # so the number of independent symlegs remains the same 
       # during transformations.
       #
       # Count the number of dependent symlegs
       num_dep_symlegs = 0
       for mp in self.maps:
           maplegs = mp.legs
           if  maplegs.issubset(self.symlegs):
               num_dep_symlegs += 1

       # Get the number of independent symlegs
       num_ind_symlegs = len(self.symlegs) - num_dep_symlegs
       return num_ind_symlegs


   @property
   def symlegs(self):
       return self._symlegs

   @property
   def maps(self):
       return self._maps

   @property
   def legs(self):
       return self._legs 

   @property 
   def maplegs(self):
       return self._maplegs 

   @property
   def nodes(self):
       return util.sort_by_legs(self._nodes)

   @property
   def num_nodes(self):
       return len(self._nodes)

   @property
   def num_out_symlegs(self):
       return self._num_out_symlegs


   def get_nodes(self, layer=None):

       if  ISNOT(layer):
           return self.nodes

       nodes = [nd for nd in self.nodes if nd.layer == layer]
       return util.sort_by_legs(nodes)


   def add_node(self, *args, **kwargs):

       node = Node(*args, **kwargs)
       legs = node.legs

       self._nodes.append(node)
       self._legs.append(legs.sorted())
       self._legs = util.sort(self._legs)
       return self


   def build(self): 

       # Build a graph of transformation nodes 

       # Add the root node to the graph (with initial symlegs)
       self._nodes = []
       self._legs  = []
       self.add_node(self.symlegs)

       # Build children of the root node (layer-1) 
       root = self.get_nodes(0)[0]
       self.build_children_of(root)

       # Build grandchildren of the root node (layer-2)
       for node in self.get_nodes(1):
           self.build_children_of(node)

       # We only build two generations of nodes (root, children, 
       # grandchildren) because there are only two generations of maps: 
       # original maps (representing tensor and aux symmetries) and 
       # contracted maps (coming from contractions of original maps).
       # Two generations of maps allow us to get from initial symlegs to any
       # any possible final symlegs within at most two generations of nodes.
       return self



   def build_children_of(self, node): 
 
       # Build children of the input "node" using a list of sorted maps
       node_legs = node.legs

       for mp in self.maps:

           # Map can introduce at most one new leg 
           # --> no valid transformation is possible 
           #
           # Example with auxiliary legs, 
           # to go from AKL with hidden B, to ABQ with hidden K: 
           # AKL --(KLQ)--> AKQ --(KBQ)--> ABQ
           map_legs = mp.legs
           if  len(map_legs - node_legs) > 1:
               continue

           # Build children of the input "node" using this particular map
           self.build_children_of_node_using_map(node, mp) 

       return self



   def build_children_of_node_using_map(self, node, mp):

       # Build children of the input "node" using a given map "mp"

       # Get combinations of node and map legs
       map_legs  = mp.legs
       node_legs = node.legs

       ALL    = node_legs | map_legs
       SHARED = node_legs & map_legs
       EXTRA  = ALL - SHARED

       # Output cannot contain dependent legs 
       # --> extra legs cannot contain dependent legs
       if  len(EXTRA) > self.num_out_symlegs:
           return

       # Shared legs fall into two groups: 
       # dot-multiplied      (disappear after contraction) and 
       # Hadamard-multiplied (remain    after contraction)
       # Output legs = extra legs + Hadamard-multiplied legs 
       num_hadamard = self.num_out_symlegs - len(EXTRA)

       # Form all possible output legs for each possible choice 
       # of Hadamard-multiplied  legs subset of shared legs 
       # (produces only one permutation of each subset)
       for HADAMARD in util.combinations(SHARED, num_hadamard): 
           
           # Get output legs (extra + Hadamard-multiplied) and dot-multiplied legs
           OUT = Str(HADAMARD) | EXTRA
           DOT = ALL - OUT
        
           # If output legs are valid, add a new node. 
           #
           # The output is invalid if: 
           #
           # -- it's already registered
           #
           # -- contains dependent legs (if output legs \in one map legs, 
           #    map node obeys a conservation law and makes one leg redundant)
           # 
           # -- independent legs are missing from the output
           #    (if more than one leg is summed out) 
           if  OUT not in self.legs    and \
               OUT not in self.maplegs and len(DOT) <= 1:

               self.add_node(OUT, mp, node)

       return self

       

   
class Node:

   def __init__(self, legs_, map_=None, previous_=None, reversed_=False): 

       if ISNOT(map_):
          msg = "Node: if map_ is None, previous_ must also be None"
          assert ISNOT(previous_), msg

       self.legs     = legs_
       self.map      = map_
       self.previous = previous_
       self.reversed = reversed_

       self.layer = 0
       if IS(self.previous):
          self.layer = self.previous.layer + 1

            
   def copy(self, reversed_=None):

       if  reversed_ is None:
           reversed_ = self.reversed
          
       node = type(self)(self.legs, self.map, self.previous, reversed_)
       return node


   def reversed_copy(self):

       reversed_ = not self.reversed 
       return self.copy(reversed_)
       

   @property
   def start_legs(self):

       if  ISNOT(self.previous):
           return self.legs

       if  self.reversed: 
           return self.legs
       return self.previous.legs


   @property
   def end_legs(self):

       if  ISNOT(self.previous):
           return self.legs

       if  self.reversed:
           return self.previous.legs
       return self.legs


   @property
   def map_legs(self): 

       if  ISNOT(self.map):
           return None
       return self.map.legs


   @property
   def map_array(self): 

       if  ISNOT(self.map):
           return None
       return self.map.array




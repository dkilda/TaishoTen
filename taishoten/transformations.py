#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy  as cp
import itertools


import taishoten.util as util
from taishoten.util import StrSet
from taishoten.util import assertequal
from taishoten.util import IS, ISNOT, ARE, ARENOT




def find_optimal_transform_path(maps, symlegs): 

    # Get symleg dims and the number of independent symlegs
    symleg_dims     = get_symleg_dims(symlegs, maps)
    num_ind_symlegs = get_num_ind_symlegs(symlegs)

    # Build complete transformation node lists for A, B, C tensors
    nodes_A = build_transform_nodes_list(symlegs["A"], maps) 
    nodes_B = build_transform_nodes_list(symlegs["B"], maps)
    nodes_C = build_transform_nodes_list(symlegs["C"], maps)
    
    # Get all combinations of nodes a, b, c 
    # (corresponding to tensors A, B, C). Try them as final nodes.
    paths = []
    costs = []

    for a, b, c in util.cartesian_prod(nodes_A, nodes_B, nodes_C):

        # Check if (a, b, c) works as a final combo of nodes 
        # whose symlegs can be directly contracted
        node_legs = [a.legs, b.legs, c.legs]
        if not good_to_contract(node_legs, num_ind_symlegs, truncated=True):
           continue

        # Compute transformations paths for A,B,C tensors: 
        # from the initial symlegs combo to the final combo (a,b,c) 
        path_A = find_transform_path(a)
        path_B = find_transform_path(b)         
        path_C = find_transform_path(c, reverse=True)    
 
        # Get cost of each transformation path and cost_AB of
        # contracting the final a, b symlegs --> add all costs
        cost_A = get_cost(path_A, symleg_dims)
        cost_B = get_cost(path_B, symleg_dims)
        cost_C = get_cost(path_C, symleg_dims)

        cost_AB = get_cost_AB(a, b, symleg_dims)
        cost    = cost_A + cost_B + cost_C + cost_AB

        # Add transformation paths for A, B, C tensors 
        # (and its cost) to the list
        paths.append((path_A, path_B, path_C))
        costs.append(cost)

     # Pick the lowest cost path
     idx  = costs.index(min(costs))
     path = paths[idx]
     return path




def get_cost_AB(a, b, symleg_dims):

    # Get cost of contracting the final a, b symlegs
    symlegs = a.end_legs | b.end_legs
    cost    = prod_of_dims(symlegs, symleg_dims)  
 
    return cost



def get_cost(path, symleg_dims):

    # Get computational cost of a given path
    cost = 0
    for node in path:
        symlegs  = node.start_legs | node.map_legs
        cost    += prod_of_dims(symlegs, symleg_dims)

    return cost


     
def prod_of_dims(legs, leg_dims):
    return np.prod([leg_dims[leg] for leg in legs])
    
    



def find_transform_path(final, reverse=False):

    # Compute a path from initial to final node 
    # in the transformation graph. As long as we take an existing
    # final node in the graph, the path will lead us to the initial node.
    path = []
    current = final

    # Starting from the final node, travel down the graph 
    # until we reach the initial node (with previous = None).
    # If we want a reversed path from final to initial node instead, 
    # make a path of reversed nodes (with start and end legs swapped).
    while current.previous:
             
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






def build_transform_nodes_list(symlegs, maps):

    # Initializes and builds transformation graph in one step
    graph = TransformGraph()
    nodes = graph.build(symlegs, maps)
 
    return nodes
    




class TransformGraph:

   def __init__(self):

       # Initialize a graph of transformation nodes
       self._num_out_symlegs = None 
       self._nodes = []
       self._legs  = []


   @property
   def nodes(self):
       return tuple(self._nodes)

   @property
   def legs(self):
       return tuple(self._legs)

   @property
   def num_nodes(self):
       return len(self._nodes)

   @property
   def num_out_symlegs(self):
       return self._num_out_symlegs


   def add_node(self, *args, **kwargs):

       node = Node(*args, **kwargs)
       self._nodes.append(node)
       self._legs.append(node.legs)
       return self



   def build(self, symlegs, maps): 

       # Truncate symlegs input
       symlegs = util.truncate(symlegs)

       # Build a graph of transformation nodes 
       self._nodes = []
       self._legs  = []
       self._num_out_symlegs = get_num_out_symlegs(symlegs, maps)

       # Add the initial, root node to the graph (with initial symlegs)
       self.add_node(symlegs, None, None)

       # Build children of the root node (first generation)
       root = self.nodes[0]
       self.build_children_of(root, maps)

       # Build grandchildren of the root node 
       # (children of the first generation nodes)
       first_generation_nodes = self.nodes[1:self.num_nodes]
       for node in first_generation_nodes:
           self.build_children_of(node, maps)

       # We only build two generations of nodes (root, children, 
       # grandchildren) because there are only two generations of maps: 
       # original maps (representing tensor and aux symmetries) and 
       # contracted maps (coming from contractions of original maps).
       # Two generations of maps allow us to get from initial symlegs to any
       # any possible final symlegs within at most two generations of nodes.
       return self._nodes



   def build_children_of(self, node, maps): 
 
       # Build children of the input "node" using a list of maps
       node_legs = node.legs

       for mp in maps:

           # Get combinations of node and map legs
           map_legs = mp.legs

           ALL    = node_legs | map_legs
           SHARED = node_legs & map_legs
           EXTRA  = ALL - SHARED

           # Map can introduce at most one new leg 
           # --> no valid transformation is possible 
           #
           # Example with auxiliary legs, 
           # to go from AKL with hidden B, to ABQ with hidden K: 
           # AKL --(KLQ)--> AKQ --(KBQ)--> ABQ
           if  len(map_legs - node_legs) > 1:
               continue

           # Output cannot contain dependent legs 
           # --> extra legs cannot contain dependent legs
           if  len(EXTRA) > self.num_out_symlegs:
               continue

           # Build children of the input "node" using this particular map
           self.build_children_of_node_using_map(node, mp, \
                                                 ALL, SHARED, EXTRA) 
       return self




   def build_children_of_node_using_map(self, node, mp, ALL, SHARED, EXTRA):

       # Build children of the input "node" using a given map "mp"
       map_legs = mp.legs

       # Output legs = extra legs + subset of shared legs 
       # (because some shared legs are contracted away) 

       # Form all possible output legs for each possible subset of shared legs
       subset_size = self.num_out_symlegs - len(EXTRA)

       for SHARED_subset in util.combinations(SHARED, subset_size):
           
           # Get output legs and summed-out legs
           OUT    = SHARED_subset + EXTRA
           SUMMED = ALL - OUT
        
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
           if  OUT not in self.legs and \
               OUT not in map_legs  and len(SUMMED) <= 1:

               self.add_node(OUT, mp, node)

       return self

       

   




class Node:

   def __init__(self, legs_, map_, previous_, reversed_=False):

       self.legs     = legs_
       self.map      = map_
       self.previous = previous_
       self.reversed = reversed_


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

       if  self.reversed:
           return self.legs
       return self.previous.legs


   @property
   def end_legs(self):

       if  self.reversed:
           return self.previous.legs
       return self.legs


   @property
   def map_legs(self):
       return self.map.legs

   @property
   def map_array(self):
       return self.map.array

   @property 
   def map(self):
       return self.map








def good_to_contract(symlegs, num_ind_symlegs=None, truncated=False): 

    # Truncate symlegs (if not truncated already) 
    if  not truncated:
        symlegs = util.truncate(symlegs)

    # Set num of independent legs
    if  num_ind_symlegs is None:
        num_ind_symlegs = get_num_ind_symlegs(symlegs, truncated=True)

    # Input and output symlegs
    IN  = symlegs["A"] + symlegs["B"]
    OUT = symlegs["C"]

    # Two conditions for direct contractability, (1) on input, (2) on output:
    #
    # (1) num of input legs = num of independent legs: 
    #     input does not contain any dependent   legs 
    #       and does not omit    any independent legs
    # 
    # (2) output legs \in input legs: 
    #     output does not have any new legs not present in the input
    is_directly_contractable = \
                (len(IN) == num_ind_symlegs) and OUT.issubset(IN)
    return is_directly_contractable




def get_num_ind_symlegs(symlegs, truncated=False): 

    # Truncate symlegs (if not truncated already) 
    if  not truncated:
        symlegs = util.truncate(symlegs)

    # Sort symleg groups by length: must iterate groups with unseen legs 
    # from-shortest-to-longest to count the number of dependent legs correctly
    sorted_symlegs = sorted(symlegs.values(), key=len)

    # Count the dependent legs
    num_dep_symlegs = 0
    visited_symlegs = StrSet()

    for symlg in sorted_symlegs:

        # There's one dependent leg in each (group containing unvisited legs)
        if  not symlg.issubset(visited_symlegs):
            visited_symlegs = visited_symlegs | symlg
            num_dep_legs += 1
 
    # Get the number of independent symlegs
    num_symlegs     = util.get_num_legs(sorted_symlegs)
    num_ind_symlegs = num_symlegs - num_dep_symlegs
    return num_ind_symlegs
    




def get_num_out_symlegs(symlegs, maps): 

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
    for mp in maps:
        maplegs = mp.legs
        if  maplegs.issubset(symlegs):
            num_dep_symlegs += 1

    # Get the number of independent symlegs
    num_ind_symlegs = len(symlegs) - num_dep_symlegs
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


































































































































































































s

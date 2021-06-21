#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import copy  as cp
import numpy as np

import lib
import util

import taishoten as tn
import taishoten.transformations as tntra

from taishoten import Str
from taishoten.transformations import Node





def get_sorted_node_legs(nodes):
    legs = tn.util.get_legs(nodes)
    legs = tn.util.sort([l.sorted() for l in legs])
    return legs





@pytest.fixture
def fixt_node():

    np.random.seed(1) 
    mp, _ = lib.make_random_map(Str("IMNO"), (5,4,5,2), 10)

    legs1 = Str("IMO")
    legs2 = Str("INO")

    node1 = Node(legs1)
    node2 = Node(legs2, mp, node1) 
    node3 = Node(legs2, mp, node1, reversed_=True)

    legs  = (legs1, legs2)
    nodes = (node1, node2, node3)
    return nodes, legs, mp






# --- Testing Node class ---------------------------------------------------- #

class TestNode:

   @pytest.fixture(autouse=True)
   def request_node_data(self, fixt_node):

       nodes, legs, mp = fixt_node

       self.nodes = nodes
       self.legs  = legs
       self.map   = mp


   def test_construct(self):

       node1, node2, node3 = self.nodes
       legs1, legs2        = self.legs
       mp                  = self.map

       util.assert_node(node1, legs1)
       util.assert_node(node2, legs2, mp)
       util.assert_node(node3, legs2, mp, reversed_=True)


   def test_copy(self):

       node1, node2, node3 = self.nodes

       out = node2.copy()
       util.assert_node_equal(out, node2)

       out = node2.reversed_copy()
       util.assert_node_equal(out, node3)





# --- Base class for tests using transformation data fixture ---------------- #

class BaseTestTransform:

   @pytest.fixture(autouse=True)
   def request_transform_data(self, fixt_transform_data):
       self.trans = fixt_transform_data

   @property
   def maps(self):
       return self.trans.maps

   @property
   def nodes(self):
       return self.trans.nodes

   @property
   def pathlets(self):
       return self.trans.pathlets 

   @property
   def symlegs(self):
       return self.trans.symlegs

   @property
   def symleg_dims(self):
       return self.trans.symleg_dims

   @property
   def Nout(self):
       return self.trans.num_out_symlegs

   def map(self, legs):
       return self.trans.map(legs)

   def node(self, key, legs):
       return self.trans.node(key, legs)




# --- Testing TransformGraph class ------------------------------------------ #

class TestTransformGraph(BaseTestTransform):


   # --- Test constructor --------------------------------------------------- #

   @pytest.mark.parametrize("key", ["A","B","C"]) 
   def test_construct(self, key):

       symlegs = self.symlegs[key]
       nodes   = []

       out       = tntra.TransformGraph(symlegs, self.maps)
       node_legs = get_sorted_node_legs(nodes)

       util.assert_graph(out, nodes, node_legs, self.Nout)
       util.assert_list(out.maps, self.maps, fun=util.assert_map_equal)
       util.assert_Str_equal(out.symlegs, symlegs[:-1])



   @pytest.mark.parametrize("legs, ans", [[Str("NOLJ"),  3], \
                                          [Str("IJLMK"), 3], \
                                          [Str("IJLNK"), 4]])  
   def test_get_num_out_symlegs(self, legs, ans): 

       graph = tntra.TransformGraph(legs, self.maps)
       assert graph.get_num_out_symlegs() == ans  



   # --- Test add/get nodes ------------------------------------------------- #

   @pytest.mark.parametrize("key, legs1, legs2, legs12", \
                            [["A", Str("IJL"), Str("IJM"), Str("IJLM")]])  
   def test_add_node(self, key, legs1, legs2, legs12):

       # Initialize graph
       out = tntra.TransformGraph(self.symlegs[key], self.maps)

       # Make nodes
       node1 = self.node(key, legs1)
       node2 = self.node(key, legs2)

       # Test node-1
       out.add_node(legs1)
       util.assert_graph(out, [node1], [legs1], self.Nout)

       # Test node-2
       out.add_node(legs2, self.map(legs12), node1)
       util.assert_graph(out, [node1, node2], [legs1, legs2], self.Nout)




   @pytest.mark.parametrize("key, legs0, legs1, legs2",           \
   [["A",                                                         \
    [Str("IJL")],                                                 \
    [Str("IJM"), Str("ILQ"), Str("IJQ"), Str("ILM"), Str("JLM")], \
    [Str("JMQ"), Str("LMQ")]]                                     \
   ])
   def test_get_nodes(self, key, legs0, legs1, legs2):

       # Node groups
       nodes_all = self.nodes[key]
       nodes_0   = tn.util.sort_by_legs([self.node(key, l) for l in legs0])
       nodes_1   = tn.util.sort_by_legs([self.node(key, l) for l in legs1])
       nodes_2   = tn.util.sort_by_legs([self.node(key, l) for l in legs2])

       # Build graph
       graph = tntra.TransformGraph(self.symlegs[key], self.maps)

       for node in nodes_0:
           graph.add_node(node.legs, node.map, node.previous)

       for node in nodes_1:
           graph.add_node(node.legs, node.map, node.previous)

       for node in nodes_2:
           graph.add_node(node.legs, node.map, node.previous)

       # Test graph
       out = graph.get_nodes()
       util.assert_list(out, nodes_all, fun=util.assert_node_equal)

       out = graph.get_nodes(0)
       util.assert_list(out, nodes_0, fun=util.assert_node_equal)

       out = graph.get_nodes(1)
       util.assert_list(out, nodes_1, fun=util.assert_node_equal)

       out = graph.get_nodes(2)
       util.assert_list(out, nodes_2, fun=util.assert_node_equal)




   # --- Test graph building ------------------------------------------------ #

   @pytest.mark.parametrize("key, r1, m1, legs1, r2, m2, legs2",    \
   [["A",                                                           \
     Str("IJL"), Str("IJLM"), [Str("IJM"), Str("ILM"), Str("JLM")], \
     Str("ILM"), Str("IMQ"),  [Str("ILQ"), Str("LMQ")],             \
   ]])
   def test_build_children_of_node_using_map(self, key, r1, m1, legs1, \
                                                        r2, m2, legs2):
       # Get roots, maps, and nodes
       root1  = self.node(key, r1)
       map1   = self.map(m1)
       nodes1 = [Node(l, map1, root1) for l in legs1] + [root1] 
       nodes1 = tn.util.sort_by_legs(nodes1)

       node_legs_1 = get_sorted_node_legs(nodes1)

       root2  = self.node(key, r2)
       map2   = self.map(m2)
       nodes2 = [Node(l, map2, root2) for l in legs2] + nodes1
       nodes2 = tn.util.sort_by_legs(nodes2) 

       node_legs_2 = get_sorted_node_legs(nodes2)
 
       # Initialize graph
       graph = tntra.TransformGraph(self.symlegs[key], self.maps)
       graph.add_node(root1.legs) 

       # Test, tier-1
       graph.build_children_of_node_using_map(root1, map1)
       util.assert_graph(graph, nodes1, node_legs_1, self.Nout)

       # Test, tier-2
       graph.build_children_of_node_using_map(root2, map2)
       util.assert_graph(graph, nodes2, node_legs_2, self.Nout)




   @pytest.mark.parametrize("key, initlegs, legs, maplegs",   \
   [["A", [Str("IJL"), Str("IJM")                          ], \
          [Str("IJQ"), Str("JMQ"), Str("ILM"),  Str("JLM") ], \
          [Str("IMQ"), Str("IMQ"), Str("IJLM"), Str("IJLM")], \
   ]])
   def test_build_children_of(self, key, initlegs, legs, maplegs):

       # Get initial nodes from transformation data
       nodes = [self.node(key, l) for l in initlegs]
       root  = nodes[-1]
         
       # Initialize graph using the initial nodes above
       graph = tntra.TransformGraph(self.symlegs[key], self.maps)
       for node in nodes:
           graph.add_node(node.legs, node.map, node.previous)

       # Get other nodes
       for l, ml in zip(legs, maplegs):
           x = Node(l, self.map(ml), root)
           nodes.append(x)

       nodes = tn.util.sort_by_legs(nodes)
       node_legs = get_sorted_node_legs(nodes)
           
       # Build graph: add children of the last node
       graph.build_children_of(root)

       # Test
       util.assert_graph(graph, nodes, node_legs, self.Nout)




   @pytest.mark.parametrize("key",          ["A","B","C"]) 
   @pytest.mark.parametrize("which_method", [0,1])
   def test_build(self, key, which_method):

       nodes   = self.nodes[key] 
       symlegs = self.symlegs[key]

       def build():
           if   which_method == 0:
                graph = tntra.build_transform_graph(symlegs, \
                                                                 self.maps)
                return graph
           else:
                graph = tntra.TransformGraph(symlegs, self.maps)
                graph.build()
                return graph

       out = build() 
       node_legs = get_sorted_node_legs(nodes)

       util.assert_graph(out, nodes, node_legs, self.Nout)





# --- Testing transformation paths ------------------------------------------ #

class TestTransformations(BaseTestTransform):

   # --- Test cost, contractability, independent symlegs, etc --------------- #

   @pytest.mark.parametrize("keyA, keyB, ans",                \
   [                                                          \
    [("A", Str("ILM")), ("A", Str("JLM")),  160],             \
    [("A", Str("ILQ")), ("B", Str("LNQ")),  lambda x: 100*x], \
    [("A", Str("ILQ")), ("B", Str("LOQ")),  lambda x: 40*x],  \
    [("A", Str("IJQ")), ("B", Str("JOQ")),  lambda x: 20*x],  \
    [("A", Str("IJQ")), ("B", Str("JNQ")),  lambda x: 50*x],  \
   ]) 
   def test_get_cost(self, keyA, keyB, ans):

       a = self.node(*keyA)
       b = self.node(*keyB)

       if  callable(ans):
           ans = ans(self.symleg_dims[Str("Q")])

       out = tntra.get_cost(a, b, self.symleg_dims)
       assert out == ans



   @pytest.mark.parametrize("legs, num_ind, ans",           \
   [                                                        \
    [(Str("IJL"),  Str("IJL"),  Str("IJL")),  None, True],  \
    [(Str("IJL"),  Str("IJK"),  Str("IJL")),  None, True],  \
    [(Str("IJK"),  Str("KLM"),  Str("IJLM")), None, False], \
    [(Str("IJLM"), Str("NOJL"), Str("INOM")), None, False], \
    [(Str("IJL"),  Str("NOJ"),  Str("INO")),  4,    False], \
    [(Str("IJ"),   Str("KL"),   Str("IJL")),  3,    False], \
    [(Str("IJQ"),  Str("JOQ"),  Str("IOQ")),  4,    True],  \
    [(Str("ILQ"),  Str("LOQ"),  Str("IOQ")),  4,    True],  \
    [(Str("IJQ"),  Str("JNQ"),  Str("INQ")),  4,    True],  \
    [(Str("ILQ"),  Str("LNQ"),  Str("INQ")),  4,    True],  \
   ]) 
   def test_good_to_contract(self, legs, num_ind, ans):

       symlegs = tn.dictriplet(*legs)

       if   num_ind:
            out = tntra.good_to_contract(symlegs, num_ind) 
       else:
            out = tntra.good_to_contract(symlegs)

       assert out == ans   



   @pytest.mark.parametrize("legs, ans",          \
   [  
    [(Str("IJLM"), Str("NOJL"), Str("INOM")), 4], \
    [(Str("IJK"),  Str("KLM"),  Str("IJLM")), 3], \
    [(Str("IJMK"), Str("IMK"),  Str("JMK")),  2], \
    [(Str("IJK"),  Str("IK"),   Str("JK")),   1], \
    [(Str("IJK"),  Str("IMK"),  Str("JMK")),  2], \
   ]) 
   def test_get_num_ind_symlegs(self, legs, ans):

       symlegs = tn.dictriplet(*legs)
       assert tntra.get_num_ind_symlegs(symlegs) == ans



   def test_get_symlegs_dims(self):

       out = tntra.get_symleg_dims(self.maps)
       ans = {l: dim for l, dim in self.symleg_dims.items()}
       util.assert_dict(out, ans)


   # --- Test transformation paths and pathlets ----------------------------- #

   @pytest.mark.parametrize("key, end_symlegs, reverse",   \
   [["A", Str("ILQ"), False], \
    ["A", Str("IJQ"), False], \
    ["B", Str("LNQ"), False], \
    ["B", Str("LOQ"), False], \
    ["B", Str("JNQ"), False], \
    ["B", Str("JOQ"), False], \
    ["C", Str("INQ"), True],  \
    ["C", Str("IOQ"), True],  \
   ])
   def test_find_transform_pathlet(self, key, end_symlegs, reverse):

       start_symlegs = self.symlegs[key]

       graph = tntra.build_transform_graph(start_symlegs, self.maps)
       final_node = tn.util.get_from_legs(graph.nodes, end_symlegs)

       out = tntra.find_transform_pathlet(final_node, reverse)
       util.assert_pathlet(out, self.maps, start_symlegs[:-1], end_symlegs, reverse)

       ans = self.pathlets[(key, end_symlegs)]


       util.assert_pathlet_equal(out, ans)



   @pytest.mark.parametrize("end_legs, num_ind", \
   [                                             \
    [[Str("JMQ"), Str("JOQ"), Str("MOQ")], 4],   \
   ])
   def test_find_transform_path(self, end_legs, num_ind):

       keys          = ("A", "B", "C")
       end_symlegs   = tn.dictriplet(*end_legs)
       start_symlegs = tn.dictriplet(*[self.symlegs[k][:-1] for k in keys])

       out, out1 = tntra.find_transform_path(self.maps, self.symlegs) 
       ans = {key: self.pathlets[(key, end_symlegs[key])] for key in keys}

       util.assert_path(out, self.maps, start_symlegs, end_symlegs, num_ind)
       util.assert_path_equal(out, ans)
       util.assert_dict(out1, end_symlegs)



































































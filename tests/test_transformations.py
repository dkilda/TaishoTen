#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import helper_lib as lib
import taishoten as tn

from taishoten import Str
from .util     import TaishoTenTestCase
from .         import util


# --- Testing transformation paths ------------------------------------------ #

class TestTransformations(TaishoTenTestCase):

   def setUp(self):

       self.trans = lib.TransformPathMaker()

      
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

   def map(self, legs):
       return self.trans.map(legs)

   def node(self, key, legs):
       return self.trans.node(key, legs)


   # --- Test cost, contractability, independent symlegs, etc --------------- #

   def test_get_cost(self):

       dimQ = self.symleg_dims["Q"]

       # Test-1
       a = tn.Node(Str("JMQ"))
       b = tn.Node(Str("JNQ")) 

       out = tn.transformations.get_cost(a, b, self.symleg_dims)
       self.assertEqual(out, 40*dimQ)

       # Test-2
       a = tn.Node(Str("ILM"))
       b = tn.Node(Str("JLN")) 

       out = tn.transformations.get_cost(a, b, self.symleg_dims)
       self.assertEqual(out, 800)

       # Test-3
       a = self.node("A", Str("ILQ"))
       b = self.node("B", Str("NLQ"))

       out = tn.transformations.get_cost(a, b, self.symleg_dims)
       self.assertEqual(out, 100*dimQ)

       # Test-4
       a = self.node("A", Str("ILQ"))
       b = self.node("B", Str("OLQ"))

       out = tn.transformations.get_cost(a, b, self.symleg_dims)
       self.assertEqual(out, 40*dimQ)

       # Test-5
       a = self.node("A", Str("IJQ"))
       b = self.node("B", Str("OJQ"))

       out = tn.transformations.get_cost(a, b, self.symleg_dims)
       self.assertEqual(out, 20*dimQ)

       # Test-6
       a = self.node("A", Str("IJQ"))
       b = self.node("B", Str("NJQ"))

       out = tn.transformations.get_cost(a, b, self.symleg_dims)
       self.assertEqual(out, 100*dimQ)



   def test_good_to_contract(self):

       # Test-1
       symlegs = tn.dictriplet(Str("IJLM"), Str("NOJL"), Str("INOM"))
       assert not tn.good_to_contract(symlegs) # Nind = 6 - 2 = 4       

       symlegs = tn.dictriplet(Str("IJL"), Str("NOJ"), Str("INO"))
       assert not tn.good_to_contract(symlegs, num_ind_symlegs=4, truncated=True) 

       # Test-2
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), Str("IJLM"))
       assert not tn.good_to_contract(symlegs) # Nind = 5 - 2 = 3

       symlegs = tn.dictriplet(Str("IJ"), Str("KL"), Str("IJL"))
       assert not tn.good_to_contract(symlegs, num_ind_symlegs=3, truncated=True) 

       # Test-3
       symlegs = tn.dictriplet(Str("IJMK"), Str("IMK"), Str("JMK"))
       assert tn.good_to_contract(symlegs) # Nind = 4 - 1 = 3

       symlegs = tn.dictriplet(Str("IJM"), Str("IM"), Str("JM"))
       assert tn.good_to_contract(symlegs, num_ind_symlegs=3, truncated=True) 

       # Test-4
       symlegs = tn.dictriplet(Str("IJK"), Str("IK"), Str("JK"))
       assert tn.good_to_contract(symlegs) # Nind = 3 - 1 = 2

       symlegs = tn.dictriplet(Str("IJ"), Str("I"), Str("J"))
       assert tn.good_to_contract(symlegs, num_ind_symlegs=2, truncated=True) 

       # Test-5
       symlegs = tn.dictriplet(Str("IJK"), Str("IMK"), Str("JMK"))
       assert tn.good_to_contract(symlegs) # Nind = 4 - 2 = 2

       symlegs = tn.dictriplet(Str("IJ"), Str("IM"), Str("JM"))
       assert tn.good_to_contract(symlegs, num_ind_symlegs=2, truncated=True) 



   def test_get_num_ind_symlegs(self):

       # Test-1
       symlegs = tn.dictriplet(Str("IJLM"), Str("NOJL"), Str("INOM"))
       assert tn.get_num_ind_symlegs(symlegs) == 4  # Nind = 6 - 2 = 4

       # Test-2
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), Str("IJLM"))
       assert tn.get_num_ind_symlegs(symlegs) == 3  # Nind = 5 - 2 = 3

       # Test-3
       symlegs = tn.dictriplet(Str("IJMK"), Str("IMK"), Str("JMK"))
       assert tn.get_num_ind_symlegs(symlegs) == 3  # Nind = 4 - 1 = 3

       # Test-4
       symlegs = tn.dictriplet(Str("IJK"), Str("IK"), Str("JK"))
       assert tn.get_num_ind_symlegs(symlegs) == 2  # Nind = 3 - 1 = 2

       # Test-5
       symlegs = tn.dictriplet(Str("IJK"), Str("IMK"), Str("JMK"))     
       assert tn.get_num_ind_symlegs(symlegs) == 2  # Nind = 4 - 2 = 2



   def test_get_symlegs_dims(self):

       out = tn.transformations.get_symlegs_dims(self.maps)
       ans = self.symleg_dims
       self.assertEqualDict(out, ans)



   # --- Test transformation paths and pathlets ----------------------------- #

   def test_find_transform_pathlet(self):

       def _test(key, end_symlegs, reverse=False):

           start_symlegs = self.symlegs(key)[:-1]

           graph      = tn.build_transform_graph(start_symlegs, self.maps)
           final_node = tn.get_from_legs(graph.nodes, end_symlegs)

           out = tn.find_tranform_pathlet(final_node, reverse)
           self.assertPathlet(out, self.maps, start_symlegs, end_symlegs, reverse)

           ans = self.pathlets[(key, end_symlegs)]
           self.assertEqualPathlet(out, ans)

       # Tests for A,B,C pathlets
       _test("A", Str("ILQ"))
       _test("A", Str("IJQ"))

       _test("B", Str("LNQ"))
       _test("B", Str("LOQ"))
       _test("B", Str("JNQ"))
       _test("B", Str("JOQ"))

       _test("C", Str("INQ"), reverse=True)
       _test("C", Str("IOQ"), reverse=True)



   def test_find_transform_path(self):

       end_symlegs = tn.dictriplet(Str("IJQ"), Str("JOQ"), Str("IOQ"))

       out = tn.find_transform_path(self.maps, self.symlegs)
       ans = {key: self.pathlets[(key, end_symlegs[key])] \
                                 for key in ("A", "B", "C")}

       self.assertPath(out, self.maps, self.symlegs, end_symlegs)
       self.assertEqualPath(out, ans)





# --- Testing TransformGraph class ------------------------------------------ #

class TestTransformGraph(TaishoTenTestCase):

   def setUp(self):

       self.trans = lib.TransformPathMaker()


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

   def map(self, legs):
       return self.trans.map(legs)

   def node(self, key, legs):
       return self.trans.node(key, legs)



   # --- Test constructor --------------------------------------------------- #

   def test_construct(self):

       for key in ("A", "B", "C"):

           symlegs = self.symlegs[key]
           nodes   = self.nodes[key]
           Nout    = 3

           out = tn.TransformGraph(symlegs, self.maps)
           self.assertGraph(out, nodes, tn.get_legs(nodes), Nout)

           self.assertList(out.maps, self.maps, fun=self.assertEqualMap)
           self.assertDict(out.symlegs, self.symlegs)



   def test_get_num_out_symlegs(self): 

       graph = tn.TransformGraph(Str("NOLJ"), self.maps)
       assert graph.get_num_out_symlegs() == 3

       graph = tn.TransformGraph(Str("IJLMK"), self.maps)
       assert graph.get_num_out_symlegs() == 2



   # --- Test add/get nodes ------------------------------------------------- #

   def test_add_node(self):

       # Initialize graph
       graph = tn.TransformGraph(self.symlegs["A"], self.maps)

       # Make nodes
       node1 = tn.Node(Str("IJL"))
       node2 = tn.Node(Str("IJM"), self.map(Str("IJLM")), node1)

       # Test-1
       graph.add_node(Str("IJL"))
       self.assertGraph(out, [node1], [node1.legs], 3)

       # Test-2
       graph.add_node(Str("IJM"), self.map(Str("IJLM")), node1)
       self.assertGraph(out, [node1, node2], [node1.legs, node2.legs], 3)



   def test_get_nodes(self):

       # Create nodes
       nodes    = [None]*7
       nodes[0] = tn.Node(Str("IJL"))
       nodes[1] = tn.Node(Str("IJM"), self.map(Str("IJLM")), nodes[0])
       nodes[2] = tn.Node(Str("ILQ"), self.map(Str("JLQ")),  nodes[0])
       nodes[3] = tn.Node(Str("IJQ"), self.map(Str("IMQ")),  nodes[1])
       nodes[4] = tn.Node(Str("JMQ"), self.map(Str("IMQ")),  nodes[1])
       nodes[5] = tn.Node(Str("ILM"), self.map(Str("IJLM")), nodes[1])
       nodes[6] = tn.Node(Str("JLM"), self.map(Str("IJLM")), nodes[1])

       nodes_all = tn.sort_by_legs(nodes)
       nodes_0   = tn.sort_by_legs([nodes[0]])
       nodes_1   = tn.sort_by_legs(nodes[1:3])
       nodes_2   = tn.sort_by_legs(nodes[3:7])

       # Setup graph
       graph = tn.TransformGraph(self.symlegs["A"], self.maps)

       graph.add_node(Str("IJL"))
       graph.add_node(Str("IJM"), self.map(Str("IJLM")), nodes[0])
       graph.add_node(Str("ILQ"), self.map(Str("JLQ")),  nodes[0])
       graph.add_node(Str("IJQ"), self.map(Str("IMQ")),  nodes[1])
       graph.add_node(Str("JMQ"), self.map(Str("IMQ")),  nodes[1])
       graph.add_node(Str("ILM"), self.map(Str("IJLM")), nodes[1])
       graph.add_node(Str("JLM"), self.map(Str("IJLM")), nodes[1])

       out = graph.get_nodes()
       self.assertList(out, nodes_all, fun=self.assertEqualNode)

       out = graph.get_nodes(0)
       self.assertList(out, nodes_0, fun=self.assertEqualNode)

       out = graph.get_nodes(1)
       self.assertList(out, nodes_1, fun=self.assertEqualNode)

       out = graph.get_nodes(2)
       self.assertList(out, nodes_2, fun=self.assertEqualNode)



   # --- Test graph building ------------------------------------------------ #

   def test_build_children_of_node_using_map(self):

       # Initialize graph
       graph = tn.TransformGraph(self.symlegs["A"], self.maps)
       graph.add_node(Str("IJL"))
       
       # Test-1
       nodes    = [None]*4
       nodes[0] = tn.Node(Str("IJL"))
       nodes[1] = tn.Node(Str("IJM"), self.map(Str("IJLM")), nodes[0])
       nodes[2] = tn.Node(Str("ILM"), self.map(Str("IJLM")), nodes[0])
       nodes[3] = tn.Node(Str("JLM"), self.map(Str("IJLM")), nodes[0])
       nodes    = tn.sort_by_legs(nodes)

       root = graph.get_nodes(0)[0]
       graph.build_children_of_node_using_map(root, self.map(Str("IJLM")))

       self.assertGraph(graph, nodes, tn.get_legs(nodes), 3)

       # Test-2
       nodes   += [None]*2
       nodes[4] = tn.Node(Str("JMQ"), self.map(Str("IMQ")), nodes[2])
       nodes[5] = tn.Node(Str("LMQ"), self.map(Str("IMQ")), nodes[2])
       nodes    = tn.sort_by_legs(nodes)

       root = tn.get_from_legs(graph.get_nodes(1), Str("ILM"))
       graph.build_children_of_node_using_map(root, self.map(Str("IMQ")))

       self.assertGraph(graph, nodes, tn.get_legs(nodes), 3)



   def test_build_children_of(self):

       # Initialize graph
       graph = tn.TransformGraph(self.symlegs["A"], self.maps)
       graph.add_node(Str("IJL"))
       graph.add_node(Str("IJM"), self.map(Str("IJLM"), graph.nodes[0])

       # Test
       nodes    = [None]*6
       nodes[0] = tn.Node(Str("IJL"))
       nodes[1] = tn.Node(Str("IJM"), self.map(Str("IJLM")), nodes[0])
       nodes[2] = tn.Node(Str("IJQ"), self.map(Str("IMQ")),  nodes[1])
       nodes[3] = tn.Node(Str("JMQ"), self.map(Str("IMQ")),  nodes[1])
       nodes[4] = tn.Node(Str("ILM"), self.map(Str("IJLM")), nodes[1])
       nodes[5] = tn.Node(Str("JLM"), self.map(Str("IJLM")), nodes[1])
       nodes    = tn.sort_by_legs(nodes)
        
       root = graph.get_nodes(1)[0]
       graph.build_children_of(root)

       self.assertGraph(graph, nodes, tn.get_legs(nodes), 3)



   def test_build(self):

       def _test(key):

           nodes   = self.nodes[key] 
           symlegs = self.symlegs[key]

           out  = tn.build_transform_graph(symlegs, self.maps)
           out1 = tn.TransformGraph(symlegs, self.maps)
           out1.build()

           self.assertGraph(out,  nodes, tn.get_legs(nodes), 3)
           self.assertGraph(out1, nodes, tn.get_legs(nodes), 3)

       # Test
       for key in ("A", "B", "C"):
           _test(key)







# --- Testing Node class ---------------------------------------------------- #

class TestNode(TaishoTenTestCase):


   def test_construct(self):

       np.random.seed(1) 

       # Test-1
       legs  = Str("INO")
       node1 = tn.Node(legs)
       self.assertNode(node1, legs)

       # Test-2
       legs  = Str("IMO")
       mp    = lib.create_random_map(Str("IMNO"), (2,3,4), 10)
       node2 = tn.Node(legs, mp, node1) 
       self.assertNode(node2, legs, mp)

       # Test-3
       node3 = tn.Node(legs, mp, node1, reversed_=True) 
       self.assertNode(node3, legs, mp, reversed_=True)


   def test_copy(self):

       np.random.seed(1) 
 
       mp = lib.create_random_map(Str("IMNO"), (2,3,4), 10)

       node1 = tn.Node(Str("INO"))             
       node2 = tn.Node(legs, mp, node1) 
       node3 = tn.Node(legs, mp, node1, reversed_=True) 

       # Test-1
       out = node2.copy()
       self.assertEqualNode(out, node2)

       # Test-2
       out = node2.reversed_copy()
       self.assertEqualNode(out, node3)





































































































































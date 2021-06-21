#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import copy  as cp
import numpy as np
import helper_lib as lib

import util
from util import isiterable, noniterable

import taishoten as tn
from taishoten import Str
from taishoten.transformations import Node




def np_einsum_of_maps(subscript, legsC, mapA, mapB):

    arrayC = np.einsum(subscript, mapA.array, mapB.array)
    idx    = lib.find_nonzeros(arrayC)
    arrayC = lib.map_from_idx(arrayC.shape, idx, val=1.0)
    out    = tn.Map(arrayC, legsC)
    return out





def make_aux_symlabels(sym, symindices, phase=1):

    signs     = [[], []]
    symlabels = [[], []]
    qtot      = sym.qtot
    mod       = sym.mod

    for k in (0,1):
      for i in symindices[k]:
          signs[k].append(sym.signs[i])
          symlabels[k].append(sym.symlabels[i])

    for k in (0,1):
        signs[k] = ''.join(signs[k])

    aux_symlabels = lib.make_aux_symlabels(signs, symlabels, \
                                           qtot, mod, phase=phase)

    return aux_symlabels




def make_symmetry_1D(fullsigns, symlabels, qtot=0, mod=None, signs=None):

    sym  = tn.Symmetry1D(fullsigns, symlabels, qtot, mod)

    data = {"fullsigns": fullsigns, "signs": signs, \
            "symlabels": symlabels, "qtot": qtot, "mod": mod}

    return sym, data




@pytest.fixture
def symlabels1D():

    Si = np.arange(0,5) 
    Sj = np.arange(0,2)
    Sk = np.arange(1,5)
    Sl = np.arange(0,4)
    Sm = np.arange(1,5)
    Sn = np.arange(0,5)
    So = np.arange(0,2)
    Sp = np.arange(0,5)

    symlabels = [Si,Sj,Sk,Sl,Sm,Sn,So,Sp]
    symlabels = dict(zip("ijklmnop", symlabels))
    return symlabels





@pytest.fixture
def symmetries1D(symlabels1D):
    
    dct = {}
    def make(legs, fullsigns, suffix=None, **kwargs):

        symlabels = [symlabels1D[l] for l in legs if l in symlabels1D]
        symlegs   = ''.join(l       for l in legs if l in symlabels1D)
        symlegs   = symlegs.upper()

        sym, data = make_symmetry_1D(fullsigns, symlabels, **kwargs)

        key = ",".join([legs, fullsigns]) 
        if  suffix:
            key = ",".join([key, suffix]) 

        dct[key]  = (sym, data, Str(legs), Str(symlegs))

    make("ijlm", "+++-")
    make("nojl", "++--")
    make("inom", "+++-")

    return dct





@pytest.fixture
def symmetry_contractions(symmetries1D):

    dct = {}
    def make(keyA, keyB, keyC):

        symA, _, legsA, symlegsA = symmetries1D[keyA] 
        symB, _, legsB, symlegsB = symmetries1D[keyB]
        symC, _, legsC, symlegsC = symmetries1D[keyC]

        sym     = tn.dictriplet(symA,     symB,     symC)
        legs    = tn.dictriplet(legsA,    legsB,    legsC) 
        symlegs = tn.dictriplet(symlegsA, symlegsB, symlegsC) 

        symcon = tn.SymmetryContraction(symA, symB, legs)
        dct[(keyA, keyB, keyC)] = (symcon, sym, legs, symlegs)

    make("ijlm,+++-",  "nojl,++--",  "inom,+++-")

    return dct




@pytest.fixture
def maps_fixt(symmetry_contractions, symlabels1D, aux_symlabels):

    key = ("ijlm,+++-", "nojl,++--", "inom,+++-")

    # Get symmetries
    symcon, sym, legs, symlegs = symmetry_contractions[key]

    Saux      = aux_symlabels[key]
    symlabels = [symlabels1D["j"], symlabels1D["l"], Saux]
    aux_sym   = tn.Symmetry1D("++-", symlabels)

    # Compute initial maps
    mapA = tn.Map.compute(sym["A"], Str("IJLM")) 
    mapB = tn.Map.compute(sym["B"], Str("NOJL"))
    mapQ = tn.Map.compute(aux_sym,  Str("JLQ"))
  
    # Compute map contractions
    maps, legs, shapes = make_map_contractions(mapA, mapB, mapQ, Saux)

    # Get leg dims
    legdims      = {l.upper(): len(s) for l, s in symlabels1D.items()}
    legdims["Q"] = len(Saux)

    return key, maps, legs, legdims, shapes




def make_map_contractions(mapA, mapB, mapQ, Saux):

    # Compute map contractions
    mapAB = np_einsum_of_maps("IJLM,NOJL->IMNO", Str("IMNO"), mapA, mapB)
    mapAQ = np_einsum_of_maps("IJLM,JLQ->IMQ",   Str("IMQ"),  mapA, mapQ)
    mapBQ = np_einsum_of_maps("NOJL,JLQ->NOQ",   Str("NOQ"),  mapB, mapQ)

    maps = [mapA, mapB, mapQ, mapAB, mapAQ, mapBQ]
    maps = sorted(maps, key=lambda x: x.legs)

    legs   = [Str("IJLM"), \
              Str("IMNO"), \
              Str("IMQ"),  \
              Str("JLQ"),  \
              Str("NOJL"), \
              Str("NOQ")]

    shapes = [(5,2,4,4),       \
              (5,4,5,2),       \
              (5,4,len(Saux)), \
              (2,4,len(Saux)), \
              (5,2,2,4),       \
              (5,2,len(Saux))]

    return maps, legs, shapes




@pytest.fixture
def aux_symlabels(symmetry_contractions):   

    dct = {}
    def make(*key):

        symcon, sym, _, _ = symmetry_contractions[key]

        phase = -symcon.phase
        symA  = sym["A"]
        symB  = sym["B"]

        def _make(symindicesA, symindicesB):
              
            Saux_A = make_aux_symlabels(symA, symindicesA)
            Saux_B = make_aux_symlabels(symB, symindicesB, phase=phase)
            Saux   = lib.align_symlabels(Saux_A, Saux_B)

            dct[key] = Saux

        return _make

    symidxA = [(1,2), (0,3)]
    symidxB = [(2,3), (0,1)]

    make("ijlm,+++-", "nojl,++--", "inom,+++-")(symidxA, symidxB)

    return dct






class TransformData:


   def __init__(self, symmetry_contractions, maps_fixt):

       key, maps, maplegs, mapleg_dims, shapes = maps_fixt

       _, _, _, symlegs = symmetry_contractions[key]

       self.maps = tn.util.sort_by_legs(maps)
       self.symlegs = tn.dictriplet(*[Str(symlegs[k]) for k in symlegs])

       self.symleg_dims = {Str(l): dim for l, dim in mapleg_dims.items() \
                                       if Str(l) in tn.util.join(*maplegs)}
       self.make_nodes()
       self.make_pathlets()

       

   def make_nodes(self):

       # Make nodes-A
       nodes    = [None]*8
       nodes[0] = Node(Str("IJL"))

       nodes[1] = Node(Str("IJM"), self.map(Str("IJLM")), nodes[0])
       nodes[2] = Node(Str("ILM"), self.map(Str("IJLM")), nodes[0])
       nodes[3] = Node(Str("JLM"), self.map(Str("IJLM")), nodes[0])

       nodes[4] = Node(Str("IJQ"), self.map(Str("JLQ")),  nodes[0])
       nodes[5] = Node(Str("ILQ"), self.map(Str("JLQ")),  nodes[0])

       nodes[6] = Node(Str("JMQ"), self.map(Str("IMQ")),  nodes[4])
       nodes[7] = Node(Str("LMQ"), self.map(Str("IMQ")),  nodes[5])

       nodesA = tn.util.sort_by_legs(nodes)   

       # Make nodes-B
       nodes    = [None]*8
       nodes[0] = Node(Str("NOJ"))

       nodes[1] = Node(Str("JLN"), self.map(Str("NOJL")), nodes[0])
       nodes[2] = Node(Str("JLO"), self.map(Str("NOJL")), nodes[0])
       nodes[3] = Node(Str("LNO"), self.map(Str("NOJL")), nodes[0])

       nodes[4] = Node(Str("JNQ"), self.map(Str("NOQ")),  nodes[0])
       nodes[5] = Node(Str("JOQ"), self.map(Str("NOQ")),  nodes[0])

       nodes[6] = Node(Str("LNQ"), self.map(Str("JLQ")),  nodes[4])
       nodes[7] = Node(Str("LOQ"), self.map(Str("JLQ")),  nodes[5])

       nodesB = tn.util.sort_by_legs(nodes)

       # Make nodes-C
       nodes    = [None]*8
       nodes[0] = Node(Str("INO"))

       nodes[1] = Node(Str("IMN"), self.map(Str("IMNO")), nodes[0])
       nodes[2] = Node(Str("IMO"), self.map(Str("IMNO")), nodes[0])
       nodes[3] = Node(Str("MNO"), self.map(Str("IMNO")), nodes[0])

       nodes[4] = Node(Str("INQ"), self.map(Str("NOQ")),  nodes[0])
       nodes[5] = Node(Str("IOQ"), self.map(Str("NOQ")),  nodes[0])

       nodes[6] = Node(Str("MNQ"), self.map(Str("IMQ")),  nodes[4])
       nodes[7] = Node(Str("MOQ"), self.map(Str("IMQ")),  nodes[5])

       nodesC = tn.util.sort_by_legs(nodes)

       self.nodes = tn.dictriplet(nodesA, nodesB, nodesC)



   def make_pathlets(self):

       # Construct all contractable pathlets
       self.pathlets = {}

       # A: IJL -> ILQ
       node1 = Node(self.symlegs["A"][:-1])
       node2 = Node(Str("ILQ"), self.map(Str("JLQ")), node1)

       self.pathlets[("A", Str("ILQ"))] = [node2]

       # A: IJL -> IJQ
       node1 = Node(self.symlegs["A"][:-1])
       node2 = Node(Str("IJQ"), self.map(Str("JLQ")), node1)

       self.pathlets[("A", Str("IJQ"))] = [node2]

       # A: IJL -> IJM -> JMQ
       node1 = Node(self.symlegs["A"][:-1])
       node2 = Node(Str("IJM"), self.map(Str("IJLM")), node1)
       node3 = Node(Str("JMQ"), self.map(Str("IMQ")),  node2)

       self.pathlets[("A", Str("JMQ"))] = [node2, node3]

       # B: NOJ -> JLN -> LNQ
       node1 = Node(self.symlegs["B"][:-1])
       node2 = Node(Str("JLN"), self.map(Str("NOJL")), node1)
       node3 = Node(Str("LNQ"), self.map(Str("JLQ")),  node2)

       self.pathlets[("B", Str("LNQ"))] = [node2, node3]

       # B: NOJ -> JLO -> LOQ
       node1 = Node(self.symlegs["B"][:-1])
       node2 = Node(Str("JLO"), self.map(Str("NOJL")), node1)
       node3 = Node(Str("LOQ"), self.map(Str("JLQ")),  node2)

       self.pathlets[("B", Str("LOQ"))] = [node2, node3]

       # B: NOJ -> JNQ
       node1 = Node(self.symlegs["B"][:-1])
       node2 = Node(Str("JNQ"), self.map(Str("NOQ")), node1)

       self.pathlets[("B", Str("JNQ"))] = [node2]

       # B: NOJ -> JOQ
       node1 = Node(self.symlegs["B"][:-1])
       node2 = Node(Str("JOQ"), self.map(Str("NOQ")), node1)

       self.pathlets[("B", Str("JOQ"))] = [node2]

       # C: IOQ -> INO (reversed)
       node1 = Node(self.symlegs["C"][:-1],                  reversed_=True)
       node2 = Node(Str("IOQ"), self.map(Str("NOQ")), node1, reversed_=True)

       self.pathlets[("C", Str("IOQ"))] = [node2]

       # C: INQ -> INO (reversed)
       node1 = Node(self.symlegs["C"][:-1],                  reversed_=True)
       node2 = Node(Str("INQ"), self.map(Str("NOQ")), node1, reversed_=True)

       self.pathlets[("C", Str("INQ"))] = [node2]

       # C: MOQ -> IMO -> INO (reversed)
       node1 = Node(self.symlegs["C"][:-1],                   reversed_=True)
       node2 = Node(Str("IMO"), self.map(Str("IMNO")), node1, reversed_=True)
       node3 = Node(Str("MOQ"), self.map(Str("IMQ")),  node2, reversed_=True)

       self.pathlets[("C", Str("MOQ"))] = [node3, node2]

       # C: INO -> IMO -> MOQ (non-reversed)
       node1 = Node(self.symlegs["C"][:-1],                 )
       node2 = Node(Str("IMO"), self.map(Str("IMNO")), node1)
       node3 = Node(Str("MOQ"), self.map(Str("IMQ")),  node2)

       self.pathlets[("C1", Str("MOQ"))] = [node2, node3]


   def map(self, legs):
       return tn.util.get_from_legs(self.maps, legs)


   def node(self, key, legs):
       return tn.util.get_from_legs(self.nodes[key], legs)





@pytest.fixture
def transform_fixt(symmetry_contractions, maps_fixt):

    trans_data = TransformData(symmetry_contractions, maps_fixt)
    return trans_data








# --- Testing transformation paths ------------------------------------------ #

class TestTransformations:

   @pytest.fixture(autouse=True)
   def request_transform_fixt(self, transform_fixt):
       trans_data = transform_fixt
       self.trans = trans_data

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

       out = tn.transformations.get_cost(a, b, self.symleg_dims)
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
            out = tn.transformations.good_to_contract(symlegs, num_ind) 
       else:
            out = tn.transformations.good_to_contract(symlegs)

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
       assert tn.transformations.get_num_ind_symlegs(symlegs) == ans



   def test_get_symlegs_dims(self):

       out = tn.transformations.get_symleg_dims(self.maps)
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

       graph = tn.transformations.build_transform_graph(start_symlegs, self.maps)
       final_node = tn.util.get_from_legs(graph.nodes, end_symlegs)

       out = tn.transformations.find_transform_pathlet(final_node, reverse)
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

       out, out1 = tn.transformations.find_transform_path(self.maps, self.symlegs) 
       ans = {key: self.pathlets[(key, end_symlegs[key])] for key in keys}

       util.assert_path(out, self.maps, start_symlegs, end_symlegs, num_ind)
       util.assert_path_equal(out, ans)
       util.assert_dict(out1, end_symlegs)












# --- Testing TransformGraph class ------------------------------------------ #

class TestTransformGraph:

   @pytest.fixture(autouse=True)
   def request_transform_fixt(self, transform_fixt):
       trans_data = transform_fixt
       self.trans = trans_data

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

   @pytest.mark.parametrize("key", ["A","B","C"]) 
   def test_construct(self, key):

       symlegs = self.symlegs[key]
       nodes   = []
       Nout    = 3

       out = tn.transformations.TransformGraph(symlegs, self.maps)

       util.assert_graph(out, nodes, tn.util.get_legs(nodes), Nout)
       util.assert_list(out.maps, self.maps, fun=util.assert_map_equal)
       util.assert_Str_equal(out.symlegs, symlegs[:-1])



   @pytest.mark.parametrize("legs, ans", [[Str("NOLJ"),  3], \
                                          [Str("IJLMK"), 3], \
                                          [Str("IJLNK"), 4]])  
   def test_get_num_out_symlegs(self, legs, ans): 

       graph = tn.transformations.TransformGraph(legs, self.maps)
       assert graph.get_num_out_symlegs() == ans  



   # --- Test add/get nodes ------------------------------------------------- #

   @pytest.mark.parametrize("key, legs1, legs2, legs12, Nout", \
                            [["A", Str("IJL"), Str("IJM"), Str("IJLM"), 3]])  
   def test_add_node(self, key, legs1, legs2, legs12, Nout):

       # Initialize graph
       out = tn.transformations.TransformGraph(self.symlegs[key], self.maps)

       # Make nodes
       node1 = self.node(key, legs1)
       node2 = self.node(key, legs2)

       # Test node-1
       out.add_node(legs1)
       util.assert_graph(out, [node1], [legs1], Nout)

       # Test node-2
       out.add_node(legs2, self.map(legs12), node1)
       util.assert_graph(out, [node1, node2], [legs1, legs2], Nout)




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
       graph = tn.transformations.TransformGraph(self.symlegs[key], self.maps)

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

       root2  = self.node(key, r2)
       map2   = self.map(m2)
       nodes2 = [Node(l, map2, root2) for l in legs2] + nodes1
       nodes2 = tn.util.sort_by_legs(nodes2) 
 
       # Initialize graph
       graph = tn.transformations.TransformGraph(self.symlegs[key], self.maps)
       graph.add_node(root1.legs) 

       # Test, tier-1
       graph.build_children_of_node_using_map(root1, map1)
       util.assert_graph(graph, nodes1, tn.util.get_legs(nodes1), 3)

       # Test, tier-2
       graph.build_children_of_node_using_map(root2, map2)
       util.assert_graph(graph, nodes2, tn.util.get_legs(nodes2), 3)




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
       graph = tn.transformations.TransformGraph(self.symlegs[key], self.maps)
       for node in nodes:
           graph.add_node(node.legs, node.map, node.previous)

       # Get other nodes
       for l, ml in zip(legs, maplegs):
           x = Node(l, self.map(ml), root)
           nodes.append(x)

       nodes = tn.util.sort_by_legs(nodes)
           
       # Build graph: add children of the last node
       graph.build_children_of(root)

       # Test
       util.assert_graph(graph, nodes, tn.util.get_legs(nodes), 3)




   @pytest.mark.parametrize("key",          ["A","B","C"]) 
   @pytest.mark.parametrize("which_method", [0,1])
   def test_build(self, key, which_method):

       nodes   = self.nodes[key] 
       symlegs = self.symlegs[key]

       def build():
           if   which_method == 0:
                graph = tn.transformations.build_transform_graph(symlegs, \
                                                                 self.maps)
                return graph
           else:
                graph = tn.transformations.TransformGraph(symlegs, self.maps)
                graph.build()
                return graph

       out = build() 
       util.assert_graph(out, nodes, tn.util.get_legs(nodes), 3)



def get_sorted_node_legs(nodes):
    legs = tn.util.get_legs(nodes)
    legs = tn.util.sort([l.sorted() for l in legs])
    return legs



# --- Testing Node class ---------------------------------------------------- #


@pytest.fixture
def node_fixt():

    np.random.seed(1) 
    mp = lib.create_random_map(Str("IMNO"), (5,4,5,2), 10)

    legs1 = Str("IMO")
    legs2 = Str("INO")

    node1 = Node(legs1)
    node2 = Node(legs2, mp, node1) 
    node3 = Node(legs2, mp, node1, reversed_=True)

    legs  = (legs1, legs2)
    nodes = (node1, node2, node3)
    return nodes, legs, mp




class TestNode:

   @pytest.fixture(autouse=True)
   def request_node_fixt(self, node_fixt):

       nodes, legs, mp = node_fixt

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



























































































































































































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

    Sa = np.arange(0,3)
    Sb = np.arange(0,3)
    Sc = np.arange(0,3)
    Sd = np.arange(0,3)

    symlabels = [Si,Sj,Sk,Sl,Sm,Sn,So,Sp,Sa,Sb,Sc,Sd]
    symlabels = dict(zip("ijklmnopabcd", symlabels))
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


    make("ijk", "++-")
    make("ijk", "++-", suffix="M2",            mod=2)
    make("ijk", "++-", suffix="M3",            mod=3)
    make("ijk", "++-", suffix="Q1,M3", qtot=1, mod=3)

    make("klm", "++-")
    make("klm", "++-", suffix="M3",            mod=3)
    make("klm", "++-", suffix="M5",            mod=5)
    make("klm", "++-", suffix="Q2,M3", qtot=2, mod=3)
    make("klm", "-+-", suffix="Q2,M3", qtot=2, mod=3)

    make("ijlm", "+++-")
    make("ijlm", "+++-", suffix="M2",            mod=2)
    make("ijlm", "+++-", suffix="M3",            mod=3)
    make("ijlm", "+++-", suffix="M5",            mod=5)
    make("ijlm", "+++-", suffix="Q3,M3", qtot=3, mod=3)
    make("ijlm", "--+-", suffix="Q1,M3", qtot=1, mod=3)

    make("inp",   "+--")
    make("inp",   "++-")
    make("inzxp", "++00-", signs="++-")

    make("pom",   "--+")
    make("pyzom", "-00-+", signs="--+")

    make("nojl",   "++--")
    make("nojl",   "++--",   suffix="M2", mod=2)

    make("inom",   "+++-")
    make("inom",   "+++-",   suffix="M2", mod=2)
    make("ixnyom", "+0+0+-", signs="+++-")

    make("ijkl",   "++--")
    make("jklmn",  "-++--")

    make("ijcdx",  "++--0", signs="++--")
    make("ijcxd",  "++-0-", signs="++--")
    make("abcxd",  "++-0-", signs="++--")

    return dct




def make_tensor(dense_shape, sym=None):

    shape = dense_shape

    if  sym is not None:
        sym_shape = (len(s) for s in sym.symlabels[:-1])
        shape     = (*sym_shape, *shape)

    array  = lib.randn(*shape)
    tensor = tn.Tensor.create(array, sym)

    data = {"array": tensor.array, "sym": tensor.sym}
    return tensor, data

    



@pytest.fixture
def tensors_fixt(symmetries1D):

    dct = {}
    def make(dense_shape, symkey=None):

        key = "(" + ",".join(str(s) for s in dense_shape) + ")"

        if   symkey is not None:

             sym, _, legs, symlegs = symmetries1D[symkey]
             key                   = ",".join([key, symkey])

        else:
             sym     = None
             legs    = Str('abcdefghijklmnoprstuvwxyz'[:len(dense_shape)])
             symlegs = Str("")

        tensor, data    = make_tensor(dense_shape, sym)
        data["legs"]    = legs
        data["symlegs"] = symlegs

        dct[key] = (tensor, data)

    d  = 2
    d1 = 3
    d2 = 4
    np.random.seed(1)

    make((d,d,d,d))

    make((d,d,d),   "ijk,++-")
    make((d,d,d),   "klm,++-")
    make((d,d,d,d), "ijlm,+++-")
    make((d,d,d,d), "nojl,++--")

    make((d,   d,   d,d), "inom,+++-")
    make((d,d1,d,d1,d,d), "ixnyom,+0+0+-")

    make((d,d,      d), "inp,++-")
    make((d,d,d1,d1,d), "inzxp,++00-")

    make((d,      d,d), "pom,--+")
    make((d,d1,d1,d,d), "pyzom,-00-+")

    make((d,d,d,d),   "ijkl,++--")
    make((d,d,d,d,d), "jklmn,-++--")

    make((d1,d1,d,d,d2), "ijcdx,++--0")
    make((d1,d1,d,d2,d), "ijcxd,++-0-")
    make((d,d,d,d2,d),   "abcxd,++-0-")

    return dct




###############################################################################
###############################################################################
###############################################################################


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












###############################################################################
###############################################################################
###############################################################################



    




class TestTensor:

   @pytest.fixture(autouse=True)
   def request_tensors(self, tensors_fixt):
       self._tensors_and_data = tensors_fixt


   def tensors_and_data(self, key):
       tensor, data = self._tensors_and_data[key]
       return tensor, data


   def tensors(self, key):
       tensor, _ = self._tensors_and_data[key]
       return tensor


   @pytest.fixture(autouse=True)
   def request_transform_fixt(self, transform_fixt):
       trans_data = transform_fixt
       self.trans = trans_data


   # --- Test construction -------------------------------------------------- #

   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_construct(self, key):

       tensor, data = self.tensors_and_data(key)
       array        = data["array"]
       sym          = data["sym"]

       out = tn.Tensor(array, sym)
       util.assert_tensor_vs_array(out, array, sym)



   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_create(self, key):

       out, data = self.tensors_and_data(key)
       util.assert_tensor_vs_array(out, data["array"], data["sym"])




   @pytest.mark.parametrize("key, key1",           \
   [                                               \
   ["(2,2,2,2)",           "(2,2,2,2),nojl,++--"], \
   ["(2,2,2),ijk,++-",     "(2,2,2,2),nojl,++--"], \
   ["(2,2,2,2),inom,+++-", "(2,2,2,2),nojl,++--"], \
   ])
   @pytest.mark.xfail
   def test_create_failed(self, key, key1):

       tensor,  data  = self.tensors_and_data(key)
       tensor1, data1 = self.tensors_and_data(key1)

       out = tn.Tensor.create(data["array"], data1["sym"])
        



   @pytest.mark.parametrize("key, key1",                   \
   [                                                       \
   ["(2,2,2,2)",           "(2,2,2),ijk,++-"],             \
   ["(2,2,2),ijk,++-",     "(2,2,2,2),nojl,++--"],         \
   ["(2,2,2,2),nojl,++--", "(2,2,2,2),inom,+++-"],         \
   ["(2,2,2,2),nojl,++--", "(2,3,2,3,2,2),ixnyom,+0+0+-"], \
   ])
   def test_as_new(self, key, key1):

       tensor,  data  = self.tensors_and_data(key)
       tensor1, data1 = self.tensors_and_data(key1)

       out = tensor.as_new()
       util.assert_tensor_equal(out, tensor)

       out = tensor.as_new(array=data1["array"], sym=data1["sym"])
       util.assert_tensor_equal(out, tensor1)




   # --- Test get methods --------------------------------------------------- #


   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_get_dense_dims(self, key):

       tensor, data = self.tensors_and_data(key)
       array        = data["array"]
       legs         = data["legs"]
       dims         = array.shape[-len(legs) : ]
       
       out = tensor.get_dense_dims(legs)
       ans = dict(zip(legs, dims))
       assert out == ans




   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_get_map(self, key):

       tensor, data = self.tensors_and_data(key)
       sym          = data["sym"]
       symlegs      = data["symlegs"] 

       out = tensor.get_map(symlegs)

       if   sym is None:
            assert out is None
       else:
            ans = tn.Map.compute(sym, symlegs) 
            util.assert_map_equal(out, ans)
       
       


   # --- Test auxiliary methods --------------------------------------------- #

   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \

   ])
   def test_symmetrize(self, key):

       tensor, data = self.tensors_and_data(key)
       array        = cp.deepcopy(data["array"])
       sym          = data["sym"]
       legs         = data["legs"]
       symlegs      = data["symlegs"]       

       # De-symmetrize array
       mp        = tn.Map.compute(sym, symlegs)
       subscript = tn.util.legs_to_subscript(mp.legs, mp.legs, mp.legs[:-1])
       mp2_array = np.einsum(subscript, mp.array, mp.array)

       idx = lib.find_zeros(mp2_array)
       idx = [np.unravel_index(i, mp2_array.shape) for i in idx]
       idx = sorted(set(idx))

       for i in idx:
           array[i] = array[i] + 1

       # Symmetrize
       out = tn.Tensor(array, sym)
       out.symmetrize()

       # Test
       util.assert_tensor_equal(out, tensor)



   @pytest.mark.parametrize("key", \
   [                               \
    "(2,2,2,2)",                   \
    "(2,2,2),ijk,++-",             \
    "(2,2,2,2),nojl,++--",         \
    "(2,2,2,2),inom,+++-",         \
    "(2,3,2,3,2,2),ixnyom,+0+0+-", \
   ])
   def test_get_full_array(self, key):

       tensor, data = self.tensors_and_data(key)
       array        = data["array"]
       sym          = data["sym"]
       legs         = data["legs"]
       symlegs      = data["symlegs"] 

       # Compute ans
       if   sym is None:
            ans = array

       else:
            mp = tn.Map.compute(sym, symlegs)

            subscript_legs = [symlegs[:-1] + legs, symlegs, symlegs + legs]
            subscript      = tn.util.legs_to_subscript(*subscript_legs)

            ans = np.einsum(subscript, array, mp.array)

       # Compute out
       out = tensor.get_full_array()

       # Test
       util.assert_array_close(out, ans)





   # --- Test transformation ------------------------------------------------ #

   @pytest.mark.parametrize("key, path_key, subs",                       \
   [                                                                     \
   ["(2,2,2,2),ijlm,+++-",  None,             []],                       \
   ["(2,2,2,2),ijlm,+++-", ("A", Str("JMQ")), ["IJLijlm,IJLM->IJMijlm",  \
                                               "IJMijlm,IMQ->JMQijlm"]], \
   ])
   def test_transform(self, key, path_key, subs):

       tensor = self.tensors(key)
       path   = self.trans.pathlets[path_key] if path_key is not None else []
       array  = tensor.array
       sym    = tensor.sym

       # Compute ans
       for i, sub in enumerate(subs):
           array = np.einsum(sub, array, path[i].map_array)

       ans = tn.Tensor(array, sym)

       # Compute out
       out  = tn.tensor.transform(tensor, path)
       out1 = tensor.transform(path)

       # Test
       util.assert_tensor_equal(out,  ans)
       util.assert_tensor_equal(out1, ans)




   @pytest.mark.parametrize("key, key_fw, subs_fw, key_bw, subs_bw",       \
   [                                                                       \
   ["(2,2,2,2),inom,+++-", ("C1", Str("MOQ")), ["INOinom,IMNO->IMOinom",   \
                                                "IMOinom,IMQ->MOQinom"],   \
                           ("C",  Str("MOQ")), ["MOQinom,IMQ->IMOinom",    \
                                                "IMOinom,IMNO->INOinom"]], \
   ])
   def test_transform_reverse(self, key, key_fw, subs_fw, \
                                         key_bw, subs_bw  ):

       tensor  = self.tensors(key)
       path_fw = self.trans.pathlets[key_fw] 
       path_bw = self.trans.pathlets[key_bw] 
       array   = tensor.array
       sym     = tensor.sym

       # Forward transformation
       for i, sub in enumerate(subs_fw):
           array = np.einsum(sub, array, path_fw[i].map_array)
 
       tensor = tn.Tensor(array, sym)

       # Compute ans
       for i, sub in enumerate(subs_bw):
           array = np.einsum(sub, array, path_bw[i].map_array)

       ans = tn.Tensor(array, sym)

       # Compute out
       out  = tn.tensor.transform(tensor, path_bw)
       out1 = tensor.transform(path_bw)

       # Test
       util.assert_tensor_equal(out,  ans)
       util.assert_tensor_equal(out1, ans)










@pytest.fixture
def symeinsum_fixt(tensors_fixt):

    dct = {}
    def make(keyA, keyB, sub, sub1, sub2, idx=1):

        A, _ = tensors_fixt[keyA]
        B, _ = tensors_fixt[keyB]

        dct[(sub, idx)] = (A, B, sub, sub1, sub2)


    # Mult (3,3->4)
    sub  = "ijk,klm->ijlm"
    sub1 = "IJKijk,KLMklm->IJLMijlm"
    sub2 = "IJLMijlm,IJLM->IJLijlm"

    make("(2,2,2),ijk,++-", "(2,2,2),klm,++-", sub, sub1, sub2)


    # Mult (4,4->4)
    sub  = "ijlm,nojl->inom"
    sub1 = "IJLMijlm,NOJLnojl->INOMinom"
    sub2 = "INOMinom,INOM->INOinom"

    make("(2,2,2,2),ijlm,+++-", "(2,2,2,2),nojl,++--", sub, sub1, sub2)


    # Mult (4,3->3)
    sub  = "inom,inp->pom"
    sub1 = "INOMinom,INPinp->POMpom"
    sub2 = "POMpom,POM->POpom"

    make("(2,2,2,2),inom,+++-", "(2,2,2),inp,++-", sub, sub1, sub2)


    # Mult with unsigned legs (4,3->3), 1 
    sub  = "ixnyom,inp->pxyom"
    sub1 = "INOMixnyom,INPinp->POMpxyom"
    sub2 = "POMpxyom,POM->POpxyom"

    make("(2,3,2,3,2,2),ixnyom,+0+0+-", "(2,2,2),inp,++-", sub, sub1, sub2)


    # Mult with unsigned legs (4,3->3), 2
    sub  = "ixnyom,inzxp->pyzom"
    sub1 = "INOMixnyom,INPinzxp->POMpyzom"
    sub2 = "POMpyzom,POM->POpyzom"

    make("(2,3,2,3,2,2),ixnyom,+0+0+-", "(2,2,3,3,2),inzxp,++00-", sub, sub1, sub2)


    # Mult (4,5->3)
    sub  = "ijkl,jklmn->imn"
    sub1 = "IJKLijkl,JKLMNjklmn->IMNimn"
    sub2 = "IMNimn,IMN->IMimn"

    make("(2,2,2,2),ijkl,++--", "(2,2,2,2,2),jklmn,-++--", sub, sub1, sub2)


    # Kronecker with unsigned legs (5,5->5)
    sub  = "ijcxd,abcyd->ijxyab"
    sub1 = "IJCDijcxd,ABCDabcyd->IJABijxyab"
    sub2 = "IJABijxyab,IJAB->IJAijxyab"

    make("(3,3,2,4,2),ijcxd,++-0-", \
         "(2,2,2,4,2),abcxd,++-0-", sub, sub1, sub2)


    # Dot with unsigned legs (5,5->4)
    sub  = "ijcxd,abcxd->ijab"
    sub1 = "IJCDijcxd,ABCDabcxd->IJABijab"
    sub2 = "IJABijab,IJAB->IJAijab"

    make("(3,3,2,4,2),ijcxd,++-0-", \
         "(2,2,2,4,2),abcxd,++-0-", sub, sub1, sub2)


    # Hadamard with unsigned legs (5,5->5)
    sub  = "ijcdx,abcxd->ijabx"
    sub1 = "IJCDijcdx,ABCDabcxd->IJABijabx"
    sub2 = "IJABijabx,IJAB->IJAijabx"

    make("(3,3,2,2,4),ijcdx,++--0", \
         "(2,2,2,4,2),abcxd,++-0-", sub, sub1, sub2)

    return dct









class TestSymeinsum:

   @pytest.fixture(autouse=True)
   def request_symeinsum(self, symeinsum_fixt):
       self._symeinsum_data = symeinsum_fixt


   def symeinsum_data(self, key):
       data = self._symeinsum_data[key]
       return data


   @pytest.mark.parametrize("key", \
   [ 
   ("ijk,klm->ijlm",       1),  \
   ("ijlm,nojl->inom",     1),  \
   ("inom,inp->pom",       1),  \
   ("ijkl,jklmn->imn",     1),  \
   ("ixnyom,inp->pxyom",   1),  \
   ("ixnyom,inzxp->pyzom", 1),  \
   ("ijcxd,abcyd->ijxyab", 1),  \
   ("ijcxd,abcxd->ijab",   1),  \
   ("ijcdx,abcxd->ijabx",  1),  \
   ])
   def test_symeinsum(self, key):
       
       A, B, sub, sub1, sub2 = self.symeinsum_data(key)
       util.assert_symeinsum(A, B, sub, sub1, sub2)  





# ************************ 3D SYMEINSUM ******************************************************** #



def make_symmetry_3D(fullsigns, symlabels, qtot=0, mod=None, signs=None):

    sym  = tn.Symmetry3D(fullsigns, symlabels, qtot, mod)

    data = {"fullsigns": fullsigns, "signs": signs, \
            "symlabels": symlabels, "qtot":  qtot,  "mod": mod, "ndim": 3}

    return sym, data




@pytest.fixture
def mod3D():
    mod  = (2 * np.pi / 5) * np.eye(3)
    return mod




@pytest.fixture
def symlabels3D(mod3D):

    dct = {}
    def make(mod3D, key):
        dct[key] = lib.make_symlabels_3D(mod3D, key)

    make(mod3D, (3,3,1))
    make(mod3D, (2,2,1))

    return dct




@pytest.fixture
def symmetries3D(symlabels3D, mod3D):

    kpts = symlabels3D[(2,2,1)]
    mod  = mod3D 
    
    dct = {}
    def make(legs, fullsigns, suffix=None, **kwargs):

        symlegs = ''.join(l for i,l in enumerate(legs) if fullsigns[i] != '0')
        symlegs = symlegs.upper()
        symlabels = [kpts]*len(symlegs)  

        sym, data = make_symmetry_3D(fullsigns, symlabels, **kwargs)

        key = ",".join([legs, fullsigns]) 
        if  suffix:
            key = ",".join([key, suffix]) 

        dct[key]  = (sym, data, Str(legs), Str(symlegs))


    make("ij",   "+-",   suffix="M1",    mod=mod)
    make("ijkl", "+-+-", suffix="M1",    mod=mod)
    make("ijkl", "+-+-", suffix="M1",    mod=mod)
    make("ijkl", "++--", suffix="M1",    mod=mod)
    make("ijk",  "++-",  suffix="Q1,M1", mod=mod, qtot=kpts[2])
    make("i",    "+",    suffix="Q1,M1", mod=mod, qtot=kpts[2])

    return dct





    

@pytest.fixture
def tensors_fixt_3D(symmetries3D):

    dct = {}
    def make(dense_shape, symkey=None):

        key = "(" + ",".join(str(s) for s in dense_shape) + ")"

        if   symkey is not None:

             sym, _, legs, symlegs = symmetries3D[symkey]
             key                   = ",".join([key, symkey])

        else:
             sym     = None
             legs    = Str('abcdefghijklmnoprstuvwxyz'[:len(dense_shape)])
             symlegs = Str("")

        tensor, data    = make_tensor(dense_shape, sym)
        data["legs"]    = legs
        data["symlegs"] = symlegs

        dct[key] = (tensor, data)

    d1 = 3
    d2 = 5
    np.random.seed(1)

    make((d1,d1),       "ij,+-,M1")
    make((d1,d2),       "ij,+-,M1")
    make((d2,d2),       "ij,+-,M1")

    make((d1,d2,d1,d1), "ijkl,+-+-,M1")
    make((d1,d2,d1,d2), "ijkl,+-+-,M1")
    make((d1,d1,d2,d2), "ijkl,++--,M1")

    make((d1,d1,d1,d1), "ijkl,++--,M1") 
    make((d1,d1,d2),    "ijk,++-,Q1,M1")
    make((d2,),         "i,+,Q1,M1")

    return dct










@pytest.fixture
def symeinsum_fixt_3D(tensors_fixt_3D):

    dct = {}
    def make(keyA, keyB, sub, sub1, sub2, idx=1):

        A, _ = tensors_fixt_3D[keyA]
        B, _ = tensors_fixt_3D[keyB]

        dct[(sub, idx)] = (A, B, sub, sub1, sub2)


    # Mult (2,2->2)
    sub  = "ki,ka->ia"
    sub1 = "KIki,KAka->IAia"
    sub2 = "IAia,IA->Iia"

    make("(3,3),ij,+-,M1", "(3,5),ij,+-,M1", sub, sub1, sub2)


    # Mult (2,2->4)
    sub  = "ia,jb->ijab"
    sub1 = "IAia,JBjb->IJABijab"
    sub2 = "IJABijab,IJAB->IJAijab"

    make("(3,3),ij,+-,M1", "(3,5),ij,+-,M1", sub, sub1, sub2)


    # Mult (4,2->4)
    sub  = "kclj,ic->klij"
    sub1 = "KCLJkclj,ICic->KLIJklij"
    sub2 = "KLIJklij,KLIJ->KLIklij"

    make("(3,5,3,3),ijkl,+-+-,M1", "(3,5),ij,+-,M1", sub, sub1, sub2)


    # Mult (4,4->2)
    sub  = "kcld,ilcd->ki"
    sub1 = "KCLDkcld,ILCDilcd->KIki"
    sub2 = "KIki,KI->Kki"

    make("(3,5,3,5),ijkl,+-+-,M1", \
         "(3,3,5,5),ijkl,++--,M1", sub, sub1, sub2)


    # Mult (4,4->4)
    sub  = "kcld,ijcd->klij"
    sub1 = "KCLDkcld,IJCDijcd->KLIJklij"
    sub2 = "KLIJklij,KLIJ->KLIklij"

    make("(3,5,3,5),ijkl,+-+-,M1", \
         "(3,3,5,5),ijkl,++--,M1", sub, sub1, sub2)


    # Mult (4,3->3)
    sub  = "klij,klb->ijb"
    sub1 = "KLIJklij,KLBklb->IJBijb"
    sub2 = "IJBijb,IJB->IJijb"

    make("(3,3,3,3),ijkl,++--,M1", "(3,3,5),ijk,++-,Q1,M1", sub, sub1, sub2)


    # Mult (4,3->1)
    sub  = "lkdc,kld->c"
    sub1 = "LKDClkdc,KLDkld->Cc"
    sub2 = "Cc,C->c"

    make("(3,3,5,5),ijkl,++--,M1", "(3,3,5),ijk,++-,Q1,M1", sub, sub1, sub2)


    # Mult (2,3->3)
    sub  = "bd,ijd->ijb"
    sub1 = "BDbd,IJDijd->IJBijb"
    sub2 = "IJBijb,IJB->IJijb"

    make("(5,5),ij,+-,M1", "(3,3,5),ijk,++-,Q1,M1", sub, sub1, sub2)


    # Mult (1,4->3)
    sub  = "c,ijcb->ijb"
    sub1 = "Cc,IJCBijcb->IJBijb"
    sub2 = "IJBijb,IJB->IJijb"

    make("(5),i,+,Q1,M1", "(3,3,5,5),ijkl,++--,M1", sub, sub1, sub2)


    # Mult (4,4->0)
    sub  = "ijab,ijba->"
    sub1 = "IJABijab,IJBAijba->"
    sub2 = ""

    make("(3,3,5,5),ijkl,++--,M1", "(3,3,5,5),ijkl,++--,M1", sub, sub1, sub2)
    return dct







class TestSymeinsum3D:

   @pytest.fixture(autouse=True)
   def request_symeinsum(self, symeinsum_fixt_3D):
       self._symeinsum_data = symeinsum_fixt_3D


   def symeinsum_data(self, key):
       data = self._symeinsum_data[key]
       return data



   @pytest.mark.parametrize("key", \
   [ 
   ("ki,ka->ia",       1), \
   ("ia,jb->ijab",     1), \
   ("kclj,ic->klij",   1), \
   ("kcld,ilcd->ki",   1), \
   ("kcld,ijcd->klij", 1), \
   ("klij,klb->ijb",   1), \
   ("lkdc,kld->c",     1), \
   ("bd,ijd->ijb",     1), \
   ("c,ijcb->ijb",     1), \
   ])
   def test_symeinsum(self, key):
       
       A, B, sub, sub1, sub2 = self.symeinsum_data(key)
       util.assert_symeinsum(A, B, sub, sub1, sub2)


   @pytest.mark.parametrize("key", \
   [ 
   ("ijab,ijba->",     1), \
   ])
   def test_symeinsum_inner_prod(self, key):

       A, B, sub, sub1, _ = self.symeinsum_data(key)

       out = tn.symeinsum(sub, A, B) 

       arrayA = A.get_full_array()
       arrayB = B.get_full_array()

       ans = np.einsum(sub1, arrayA, arrayB) 
       util.assert_array_close(out.array, ans)




"""
"""































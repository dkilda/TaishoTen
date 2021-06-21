#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'..')

import itertools
import copy  as cp
import numpy as np

import taishoten as tn
from taishoten import Str
from taishoten.transformations import Node



# --- Basic auxiliary functions --------------------------------------------- #


def randn(*args, cmplx=False):

    if   cmplx:
         return randn(*args) + 1.0j*randn(*args)
    else:
         return np.random.random(args)



def randint(low=0, high=None, size=None, dtype=int):

    return np.random.randint(low, high=high, size=size, dtype=dtype)



def find_zeros(x, tol=1e-6):
    return find_idx(x, lambda v: abs(v) < tol)



def find_nonzeros(x, tol=1e-6):
    return find_idx(x, lambda v: abs(v) > tol)



def find_idx(x, mask):

    shape = x.shape
    x     = x.ravel()

    # Find idx that satisfy mask condition
    idx = []
    for i in range(len(x)):
        if mask(x[i]):
           idx.append(i)

    return idx


def isiterable(x):
    try:
        x_iterator = iter(x)
        isiter = True
    except TypeError:
        isiter = False

    return isiter


def noniterable(x):
    return not isiterable(x)




# --- Summing meshgrid and flattening --------------------------------------- #

def sum_meshgrid(*xs):

    shape = tuple([len(x) for x in xs])
    shape = shape if xs[0].ndim == 1 else (*shape, xs[0].shape[-1])
    size  = np.prod(shape)
    sgrid = np.zeros(shape) 

    if    len(xs) == 1:

          for i in range(shape[0]):
              sgrid[i] = xs[0][i]

    elif  len(xs) == 2:

          for i in range(shape[0]):
            for j in range(shape[1]):

                sgrid[i,j] = xs[0][i] + xs[1][j]

    elif  len(xs) == 3:

          for i in range(shape[0]):
            for j in range(shape[1]):
              for k in range(shape[2]):

                  sgrid[i,j,k] = xs[0][i] + xs[1][j] + xs[2][k]

    elif  len(xs) == 4:

          for i in range(shape[0]):
            for j in range(shape[1]):
              for k in range(shape[2]):
                for l in range(shape[3]):

                    sgrid[i,j,k,l] = xs[0][i] + xs[1][j] \
                                   + xs[2][k] + xs[3][l]

    elif  len(xs) == 5:

          for i in range(shape[0]):
            for j in range(shape[1]):
              for k in range(shape[2]):
                for l in range(shape[3]):
                  for m in range(shape[4]):

                      sgrid[i,j,k,l,m] = xs[0][i] + xs[1][j] \
                                       + xs[2][k] + xs[3][l] + xs[4][m]

    else:
          raise NotImplementedError("sum_meshgrid: len > 5 not supported")

    return sgrid



def sum_meshgrid_1(*xs):

    grid  = np.meshgrid(*xs, indexing='ij')
    sgrid = sum(grid)
    return sgrid



def flatten(x, ndim=None):

    shape = x.shape if ndim is None else x.shape[:-1]
    size  = np.prod(shape)

    idx      = np.indices(shape)
    flat_idx = np.ravel_multi_index(idx, shape)

    flat_shape = (size, ) if ndim is None else (size, ndim)
    flat_x     = np.zeros(flat_shape)

    if    len(shape) == 1:

          for i in range(shape[0]):
              flat_x[flat_idx[i]] = x[i]

    elif  len(shape) == 2:

          for i in range(shape[0]):
            for j in range(shape[1]):

                flat_x[flat_idx[i,j]] = x[i,j]

    elif  len(shape) == 3:

          for i in range(shape[0]):
            for j in range(shape[1]):
              for k in range(shape[2]):

                  flat_x[flat_idx[i,j,k]] = x[i,j,k]

    elif  len(shape) == 4:

          for i in range(shape[0]):
            for j in range(shape[1]):
              for k in range(shape[2]):
                for l in range(shape[3]):

                    flat_x[flat_idx[i,j,k,l]] = x[i,j,k,l]

    elif  len(shape) == 5:

          for i in range(shape[0]):
            for j in range(shape[1]):
              for k in range(shape[2]):
                for l in range(shape[3]):
                  for m in range(shape[4]):

                      flat_x[flat_idx[i,j,k,l,m]] = x[i,j,k,l,m]

    else:
          raise NotImplementedError("flatten: len(shape) > 5 not supported")

    return flat_x   



def flatten_1(x):
    return x.flatten()



def flatten_symlabels(signs, symlabels, phase=1, ndim=1):

    signed_symlabels = get_signed_symlabels(signs, symlabels, phase)

    sgrid = sum_meshgrid(*signed_symlabels)
    if  ndim == 1:
        sgrid = sgrid.reshape((*sgrid.shape, 1))

    flat_symlabels = flatten(sgrid, ndim)
    return flat_symlabels



def flatten_symlabels_1(signs, symlabels, phase=1):

    # Strictly ndim=1
    signed_symlabels = get_signed_symlabels(signs, symlabels, phase)

    sgrid = sum_meshgrid_1(*signed_symlabels)
    sgrid = sgrid.reshape((*sgrid.shape, 1))

    flat_symlabels = flatten(sgrid, 1)
    return flat_symlabels



def flatten_symlabels_2(signs, symlabels, phase=1):

    # Strictly ndim=1
    signed_symlabels = get_signed_symlabels(signs, symlabels, phase)
 
    sgrid = sum_meshgrid_1(*signed_symlabels)
    sgrid = sgrid.reshape((*sgrid.shape, 1))

    flat_symlabels = flatten_1(sgrid)
    flat_symlabels = flat_symlabels.reshape(len(flat_symlabels), 1)
    return flat_symlabels




# --- Symmetry helpers ------------------------------------------------------ #

def signs_to_int(signs):

    signs_int = []
    for s in signs:

        if   s == '+':
             s_int = 1

        elif s == '-':
             s_int = -1

        else:
             s_int = 0

        signs_int.append(s_int)

    return signs_int



def get_signed_symlabels(signs, symlabels, phase=1):

    signs = signs_to_int(signs)

    signed_symlabels = []
    for i in range(len(symlabels)):
        signed_symlabels.append(symlabels[i] * signs[i] * phase)

    return signed_symlabels



def sym_size(x):
    return np.prod([len(s) for s in x])



def align_symlabels(a, b, tol=1e-6):

    idx = []
    shape = (len(a), len(b))
    combo = np.zeros(shape)

    for i in range(shape[0]):
      for j in range(shape[1]):

          tmp        = abs(a[i] - b[j])
          combo[i,j] = sum(tmp) if a.ndim > 1 else tmp

          if combo[i,j] < tol:
             idx.append(i)

    aligned = np.array([a[i] for i in idx])
    return aligned




def make_aux_symlabels(sym, symindices, phase=1, tol=1e-6):

    def _make_signs(k):
        signs = [sym.signs[i] for i in symindices[k]]
        signs = ''.join(signs)
        return signs

    def _make_symlabels(k):
        symlabels = [sym.symlabels[i] for i in symindices[k]]
        return symlabels

    # Get signs and symlabels
    lhs_signs = _make_signs(0)
    rhs_signs = _make_signs(1)

    lhs_symlabels = _make_symlabels(0)
    rhs_symlabels = _make_symlabels(1)

    qtot = sym.qtot
    mod  = sym.mod

    # Determine ndim
    ndim = 1
    if  isiterable(lhs_symlabels[0][0]):
        ndim = len(lhs_symlabels[0][0])

    # Flatten out
    lhs = flatten_symlabels(lhs_signs, lhs_symlabels, phase=phase,  ndim=ndim)
    rhs = flatten_symlabels(rhs_signs, rhs_symlabels, phase=-phase, ndim=ndim)
    rhs -= qtot

    # Fold and align
    fold = {1: fold_1D, \
            3: fold_3D}[ndim]

    lhs  = fold(lhs, mod)
    rhs  = fold(rhs, mod)
    aux  = align_symlabels(lhs, rhs, tol=tol)

    return aux




def fold_1D(symlabels, mod=None):
  
    if mod is None:
       return np.unique(symlabels)

    return np.unique(np.mod(symlabels, mod))




def fold_3D(symlabels, mod=None, decimals=10):

    if mod is None:
       return np.unique(np.round_(symlabels, decimals), axis=0)

    symlabels = np.dot(symlabels, np.linalg.inv(mod))
    symlabels = symlabels - np.floor(np.round_(symlabels, decimals))

    symlabels = np.unique(np.round_(symlabels, decimals), axis=0)
    symlabels = np.dot(symlabels, mod)
    return symlabels





# --- Creating maps --------------------------------------------------------- #

def make_random_map(legs, shape, num_elems):

    array, idx = random_map_array_from_idx(shape, num_elems)
    random_map = tn.Map(array, legs)

    data = {"legs": legs, "shape": shape, \
            "idx": idx,   "array": array}

    return random_map, data




def make_map_from_sym(legs, signs, symlabels, qtot=0, mod=None, **kwargs):

    array     = map_array_from_sym(signs, symlabels, qtot, mod, **kwargs)
    irrep_map = tn.Map(array, legs)

    data = {"array": array, "legs": legs, "signs": signs, \
            "symlabels": symlabels, "qtot": qtot, "mod": mod, **kwargs}
    return irrep_map, data




def map_array_from_sym(signs, symlabels, qtot=0, mod=None, tol=1e-6):

    ndim = 1
    if  isiterable(symlabels[0][0]):
        ndim = len(symlabels[0][0])

    flat = flatten_symlabels(signs, symlabels, ndim=ndim)
    if  mod is not None:
        flat = np.mod(flat - qtot, mod)

    if  ndim > 1:
        flat = np.sum(abs(flat), -1)

    idx   = find_zeros(flat, tol=tol)  
    shape = tuple([len(s) for s in symlabels])

    out = map_array_from_idx(shape, idx) 
    return out


def map_array_from_idx(shape, idx, val=1.0):

    out = np.zeros(shape)
    for i in idx:
        out[np.unravel_index(i, shape)] = val
    return out



def random_map_array_from_idx(shape, num_idx):

    idx   = randint(0, np.prod(shape), num_idx)
    array = map_array_from_idx(shape, idx)
    return array, idx



def make_map_contractions(mapA, mapB, mapQ, Saux):

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




def np_einsum_of_maps(subscript, legsC, mapA, mapB):

    arrayC = np.einsum(subscript, mapA.array, mapB.array)
    idx    = find_nonzeros(arrayC)

    arrayC = map_array_from_idx(arrayC.shape, idx, val=1.0)
    out    = tn.Map(arrayC, legsC)
    return out




# --- Creating symmetries --------------------------------------------------- #


def make_symmetry(fullsigns, symlabels, \
                  qtot=0, mod=None, signs=None, ndim=1):

    sym = {1: tn.Symmetry1D, \
           3: tn.Symmetry3D}[ndim](fullsigns, symlabels, qtot, mod)

    data = {"fullsigns": fullsigns, "signs": signs, \
            "symlabels": symlabels, "qtot": qtot, "mod": mod, "ndim": ndim}

    return sym, data



def make_symmetry_1D(*args, **kwargs):
    return make_symmetry(*args, ndim=1, **kwargs)




def make_symmetry_3D(*args, **kwargs):
    return make_symmetry(*args, ndim=3, **kwargs)




def make_symmetry_1D_wrap(legs, fullsigns, \
                          symlabels_data, suffix=None, **kwargs):

    symlabels = [symlabels_data[l] for l in legs if l in symlabels_data]
    symlegs   = ''.join(l          for l in legs if l in symlabels_data)
    symlegs   = symlegs.upper()

    sym, data = make_symmetry_1D(fullsigns, symlabels, **kwargs)
    key       = make_sym_key(legs, fullsigns, suffix)

    out = (sym, data, Str(legs), Str(symlegs))
    return key, out




def make_symmetry_3D_wrap(legs, fullsigns, \
                          symlabels, suffix=None, **kwargs):

    symlegs   = ''.join(l for i,l in enumerate(legs) if fullsigns[i] != '0')
    symlegs   = symlegs.upper()
    symlabels = [symlabels]*len(symlegs)  

    sym, data = make_symmetry_3D(fullsigns, symlabels, **kwargs)
    key       = make_sym_key(legs, fullsigns, suffix)

    out = (sym, data, Str(legs), Str(symlegs))
    return key, out




def make_sym_key(legs, fullsigns, suffix=None):

    key = ",".join([legs, fullsigns]) 
    if  suffix:
        key = ",".join([key, suffix]) 

    return key




def make_symlabels_3D(Gvecs, num_pts):

    # Generate lists of symlabels along x,y,z axes \in (0,1)
    # --> find  all their combos (triples) using Cartesian product
    # --> gives scaled symlabels = list of all possible triples
    symlabel_arrays  = [np.arange(n) / n for n in num_pts]
    scaled_symlabels = [np.asarray(s) \
                        for s in itertools.product(*symlabel_arrays)]

    scaled_symlabels = np.asarray(scaled_symlabels)

    # Multiply scaled symlabels by Gvecs
    # scaled_symlabels \in (0,1) -> symlabels \in (0,G)
    symlabels = np.dot(scaled_symlabels, Gvecs)
    return symlabels
    



def get_reciprocal_vecs(lattice_vecs):

    # From matrix of lattice vectors, get matrix of reciprocal vectors 
    return 2 * np.pi * np.linalg.inv(lattice_vecs.T)
 



# --- Creating symmetry contractions ---------------------------------------- #

def make_symcon(symmetries_and_data, keyA, keyB, keyC):

    symA, _, legsA, symlegsA = symmetries_and_data[keyA] 
    symB, _, legsB, symlegsB = symmetries_and_data[keyB]
    symC, _, legsC, symlegsC = symmetries_and_data[keyC]

    sym     = tn.dictriplet(symA,     symB,     symC)
    legs    = tn.dictriplet(legsA,    legsB,    legsC) 
    symlegs = tn.dictriplet(symlegsA, symlegsB, symlegsC) 

    symcon = tn.SymmetryContraction(symA, symB, legs)
    return symcon, sym, legs, symlegs



# --- Creating tensors ------------------------------------------------------ #

def make_tensor(dense_shape, sym=None):

    shape = dense_shape

    if  sym is not None:
        sym_shape = (len(s) for s in sym.symlabels[:-1])
        shape     = (*sym_shape, *shape)

    array  = randn(*shape)
    tensor = tn.Tensor.create(array, sym)

    data = {"array": tensor.array, "sym": tensor.sym}
    return tensor, data




def make_tensor_wrap(dense_shape, symmetries, symkey=None):

    key = "(" + ",".join(str(s) for s in dense_shape) + ")"

    if   symkey is None:

         sym      = None
         alphabet = 'abcdefghijklmnoprstuvwxyz'
         legs     = Str(alphabet[:len(dense_shape)])
         symlegs  = Str("")

    else:
         sym, _, legs, symlegs = symmetries[symkey]
         key                   = ",".join([key, symkey])

    tensor, data    = make_tensor(dense_shape, sym)
    data["legs"]    = legs
    data["symlegs"] = symlegs

    return key, (tensor, data)





# --- Creating transformations data ----------------------------------------- #

class TransformData:

   def __init__(self, symcon_and_data, maps_and_data):

       key, maps, maplegs, mapleg_dims, shapes = maps_and_data

       _, _, _, symlegs = symcon_and_data[key]

       self.maps    = tn.util.sort_by_legs(maps)
       self.symlegs = tn.dictriplet(*[Str(symlegs[k]) for k in symlegs])

       joined_maplegs   = tn.util.join(*maplegs)
       self.symleg_dims = {Str(l): dim for l, dim in mapleg_dims.items() \
                                       if Str(l) in joined_maplegs}
       self.make_nodes()
       self.make_pathlets()
       self.make_num_out_symlegs()



   def make_num_out_symlegs(self):
       self.num_out_symlegs = 3

       

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










































































































































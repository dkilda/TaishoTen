#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'..')

import itertools
import numpy as np
import taishoten as tn

from taishoten import Str
from .util     import CLOSE_RTOL, CLOSE_ATOL
from .util     import isiterable, noniterable


# --- Auxiliary functions --------------------------------------------------- #

def randn(*args, cmplx=False):

    if   cmplx:
         return randn(*args) + 1.0j*randn(*args)
    else:
         return np.random.randn(*args)



def randint(low=0, high=None, size=None, dtype=int):

    return np.random.randint(low, high=high, size=size, dtype=dtype)



def random_map_from_idx(shape, num_idx):

    idx = randint(0, np.prod(shape), num_idx)
    return map_from_idx(shape, idx), idx



def map_from_idx(shape, idx, val=1.0):

    out = np.zeros(shape)
    for i in idx:
        out[i] = val
    return out



def map_from_sym(signs, symlabels, qtot=0, mod=None, tol=CLOSE_RTOL):

    ndim = 1
    if  isiterable(symlabels[0][0]):
        ndim = len(symlabels[0][0])

    flat = flatten_symlabels(signs, symlabels, ndim=ndim)
    if  mod is not None:
        flat = np.mod(flat - qtot, mod)

    if  ndim > 1:
        flat = np.sum(abs(flat), -1)

    idx   = find_zeros(flat, tol=tol)  
    shape = (len(s) for s in symlabels)

    return map_from_idx(shape, idx) 



def find_zeros(x, tol=CLOSE_RTOL):

    # Find idx of zeros in x
    idx = []
    for i in range(len(x)):
        if x[i] < tol:
           idx.append(i)

    return tuple(idx)



def find_nonzeros(x, tol=CLOSE_RTOL):

    # Find idx of nonzeros in x
    idx = []
    for i in range(len(x)):
        if x[i] > tol:
           idx.append(i)

    return tuple(idx)



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




# --- Meshgrid and flattening ----------------------------------------------- #

def sum_meshgrid(*xs):

    shape = (len(x) for x in xs)
    size  = np.prod(shape)
    sgrid = np.zeros(shape) 

    if    len(xs) == 1:
          pass

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

    # Construct meshgrid
    grid  = np.meshgrid(*xs, indexing='ij')
    sgrid = sum(grid)
    return sgrid



def flatten(x, xdim=None):

    shape = x.shape if xdim is None else x.shape[:-xdim]
    size  = np.prod(shape)

    idx      = np.indices(shape)
    flat_idx = np.ravel_multi_index(idx, shape)

    flat_shape = (size, ) if xdim is None else (size, xdim)
    flat_x     = np.zeros(flat_shape)

    if    len(shape) == 1:
          pass

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
    return sgrid.flatten()



def flatten_symlabels(signs, symlabels, phase=1, ndim=1):

    signed_symlabels = get_signed_symlabels(signs, symlabels, phase=1)

    sgrid = sum_meshgrid(*signed_symlabels)
    if  ndim == 1:
        sgrid = sgrid.reshape((*sgrid.shape, 1))

    flat_symlabels = flatten(sgrid, ndim)
    return flat_symlabels



def flatten_symlabels_1(signs, symlabels, phase=1):

    # Strictly xdim=1
    signed_symlabels = get_signed_symlabels(signs, symlabels, phase=1)
 
    sgrid          = sum_meshgrid_1(*signed_symlabels)
    flat_symlabels = flatten(sgrid)

    return flat_symlabels



def flatten_symlabels_2(signs, symlabels, phase=1):

    # Strictly xdim=1
    signed_symlabels = get_signed_symlabels(signs, symlabels, phase=1)
 
    sgrid          = sum_meshgrid_1(*signed_symlabels)
    flat_symlabels = flatten_1(sgrid)

    return flat_symlabels




# --- Aligned, folded and auxiliary symlabels ------------------------------- #

def align_symlabels(a, b, tol=CLOSE_RTOL):

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



def fold_1D(symlabels, mod=None):
  
    if mod is None:
       return np.unique(symlabels)

    return np.unique(np.mod(symlabels, mod))




   
def make_aux_symlabels(signs, symlabels, \
                       qtot=0, mod=None, phase=1, tol=CLOSE_RTOL):

    # Unpack
    lhs_signs,     rhs_signs     = signs
    lhs_symlabels, rhs_symlabels = symlabels

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




# --- 3D Symmetries --------------------------------------------------------- #

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
 



def fold_3D(symlabels, mod=None, decimals=10):

    if mod is None:
       return np.unique(np.round_(symlabels, decimals), axis=0)

    symlabels = np.dot(symlabels, np.linalg.inv(mod))
    symlabels = symlabels - np.floor(np.round_(symlabels, decimals))

    symlabels = np.unique(np.round_(symlabels, decimals), axis=0)
    symlabels = np.dot(symlabels, mod)
    return symlabels





# --- Intermediate datastructures setup using TaishoTen --------------------- #

def create_random_map(legs, shape, num_elems):

    array = lib.random_map_from_idx(shape, num_elems)[0]
    out = tn.Map(array, legs)   
    return out




def setup_tensor(signs, symlabels, dense_shape, *args, **kwargs): 

    sym_shape = (len(s) for s in symlabels[:-1])

    array  = lib.randn(*sym_shape, *dense_shape) 
    sym    = tn.Symmetry1D(signs, symlabels, *args, **kwargs)
    tensor = tn.Tensor(array, sym)

    return tensor, array, sym



def setup_tensor_3D(signs, symlabels, dense_shape, *args, **kwargs): 

    sym_shape = (len(s) for s in symlabels[:-1])

    array  = lib.randn(*sym_shape, *dense_shape) 
    sym    = tn.Symmetry3D(signs, symlabels, *args, **kwargs)
    tensor = tn.Tensor(array, sym)

    return tensor, array, sym




class TransformPathMaker: 

   def __init__(self):

       # Make maps and symlegs
       Si = np.arange(0,5) 
       Sj = np.arange(0,2)
       Sl = np.arange(0,4) 
       Sm = np.arange(1,5)
       Sn = np.arange(0,5)
       So = np.arange(0,2)

       symA = tn.Symmetry1D("+++-", (Si,Sj,Sl,Sm))
       symB = tn.Symmetry1D("++--", (Sn,So,Sj,Sl))
       legs = tn.dictriplet(Str("ijlm"), Str("nojl"), Str("inom"))

       symcon  = tn.compute_symmetry_contraction(symA, symB, legs)
       maps    = tn.compute_maps(symcon)
       maps    = tn.sort_by_legs(maps)
       symlegs = symcon.symlegs()

       self.maps    = maps
       self.symlegs = symlegs

       dimQ = self.map(Str("JLQ")).shape[-1]
       self.symleg_dims = {"I": 5, "J": 2, "L": 4, \
                           "M": 4, "N": 5, "O": 2, "Q": dimQ}

       # Make nodes and pathlets
       self.make_nodes()
       self.make_pathlets()

       
   def make_nodes(self):

       # Make nodes-A
       nodes    = [None]*8
       nodes[0] = tn.Node(Str("IJL"))

       nodes[1] = tn.Node(Str("IJM"), self.map(Str("IJLM")), nodes[0])
       nodes[2] = tn.Node(Str("ILM"), self.map(Str("IJLM")), nodes[0])
       nodes[3] = tn.Node(Str("JLM"), self.map(Str("IJLM")), nodes[0])

       nodes[4] = tn.Node(Str("IJQ"), self.map(Str("JLQ")),  nodes[0])
       nodes[5] = tn.Node(Str("ILQ"), self.map(Str("JLQ")),  nodes[0])

       nodes[6] = tn.Node(Str("JMQ"), self.map(Str("IMQ")),  nodes[4])
       nodes[7] = tn.Node(Str("LMQ"), self.map(Str("IMQ")),  nodes[5])

       nodesA = tn.sort_by_legs(nodes)   

       # Make nodes-B
       nodes    = [None]*8
       nodes[0] = tn.Node(Str("NOJ"))

       nodes[1] = tn.Node(Str("JLN"), self.map(Str("NOJL")), nodes[0])
       nodes[2] = tn.Node(Str("JLO"), self.map(Str("NOJL")), nodes[0])
       nodes[3] = tn.Node(Str("LNO"), self.map(Str("NOJL")), nodes[0])

       nodes[4] = tn.Node(Str("JNQ"), self.map(Str("NOQ")),  nodes[0])
       nodes[5] = tn.Node(Str("JOQ"), self.map(Str("NOQ")),  nodes[0])

       nodes[6] = tn.Node(Str("LNQ"), self.map(Str("JLQ")),  nodes[4])
       nodes[7] = tn.Node(Str("LOQ"), self.map(Str("JLQ")),  nodes[5])

       nodesB = tn.sort_by_legs(nodes)

       # Make nodes-C
       nodes    = [None]*8
       nodes[0] = tn.Node(Str("INO"))

       nodes[1] = tn.Node(Str("IMN"), self.map(Str("INOM")), nodes[0])
       nodes[2] = tn.Node(Str("IMO"), self.map(Str("INOM")), nodes[0])
       nodes[3] = tn.Node(Str("MNO"), self.map(Str("INOM")), nodes[0])

       nodes[4] = tn.Node(Str("INQ"), self.map(Str("NOQ")),  nodes[0])
       nodes[5] = tn.Node(Str("IOQ"), self.map(Str("NOQ")),  nodes[0])

       nodes[6] = tn.Node(Str("MNQ"), self.map(Str("IMQ")),  nodes[4])
       nodes[7] = tn.Node(Str("MOQ"), self.map(Str("IMQ")),  nodes[5])

       nodesC = tn.sort_by_legs(nodes)

       self.nodes = tn.dictriplet(nodesA, nodesB, nodesC)



   def make_pathlets(self):

       # Construct all contractable pathlets
       self.pathlets = {}

       # A: IJL -> ILQ
       node1 = tn.Node(self.symlegs("A"))
       node2 = tn.Node(Str("ILQ"), self.map(Str("JLQ")), node1)

       self.pathlets[("A", Str("ILQ"))] = [node2]

       # A: IJL -> IJQ
       node1 = tn.Node(self.symlegs("A"))
       node2 = tn.Node(Str("IJQ"), self.map(Str("JLQ")), node1)

       self.pathlets[("A", Str("IJQ"))] = [node2]

       # B: NOJ -> JNQ -> LNQ
       node1 = tn.Node(self.symlegs("B"))
       node2 = tn.Node(Str("JNQ"), self.map(Str("NOQ")), node1)
       node3 = tn.Node(Str("LNQ"), self.map(Str("JLQ")), node2)

       self.pathlets[("B", Str("LNQ"))] = [node2, node3]

       # B: NOJ -> JOQ -> LOQ
       node1 = tn.Node(self.symlegs("B"))
       node2 = tn.Node(Str("JOQ"), self.map(Str("NOQ")), node1)
       node3 = tn.Node(Str("LOQ"), self.map(Str("JLQ")), node2)

       self.pathlets[("B", Str("LOQ"))] = [node2, node3]

       # B: NOJ -> JNQ
       node1 = tn.Node(self.symlegs("B"))
       node2 = tn.Node(Str("JNQ"), self.map(Str("NOQ")), node1)

       self.pathlets[("B", Str("JNQ"))] = [node2]

       # B: NOJ -> JOQ
       node1 = tn.Node(self.symlegs("B"))
       node2 = tn.Node(Str("JOQ"), self.map(Str("NOQ")), node1)

       self.pathlets[("B", Str("JOQ"))] = [node2]

       # C: IOQ -> INO (reversed)
       node1 = tn.Node(self.symlegs("C"))
       node2 = tn.Node(Str("IOQ"), self.map(Str("NOQ")), node1)

       self.pathlets[("C", Str("IOQ"))] = [node2]

       # C: INQ -> INO (reversed)
       node1 = tn.Node(self.symlegs("C"))
       node2 = tn.Node(Str("INQ"), self.map(Str("NOQ")), node1)

       self.pathlets[("C", Str("INQ"))] = [node2]



   def map(self, legs):
       return tn.get_from_legs(self.maps, legs)



   def node(self, key, legs):
       return tn.get_from_legs(self.nodes[key], legs)






























































































































































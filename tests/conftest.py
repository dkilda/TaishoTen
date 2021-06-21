#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'..')

import pytest

import itertools
import copy  as cp
import numpy as np

import lib
import taishoten as tn
from taishoten import Str




# --- Symmetries and symmetry contractions, 1D ------------------------------ #

@pytest.fixture
def fixt_symlabels_1D():

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
def fixt_symmetries_1D(fixt_symlabels_1D):
    
    symlabels = fixt_symlabels_1D
    dct = {}
    def make(legs, fullsigns, suffix=None, **kwargs):

        key, out = lib.make_symmetry_1D_wrap(legs, fullsigns, \
                                             symlabels, suffix, **kwargs)
        dct[key] = out

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
    make("nojl",   "++--", suffix="M2", mod=2)

    make("inom",   "+++-")
    make("inom",   "+++-", suffix="M2", mod=2)
    make("ixnyom", "+0+0+-", signs="+++-")

    make("ijkl",   "++--")
    make("jklmn",  "-++--")

    make("ijcdx",  "++--0", signs="++--")
    make("ijcxd",  "++-0-", signs="++--")
    make("abcxd",  "++-0-", signs="++--")

    return dct




@pytest.fixture
def fixt_symcon_1D(fixt_symmetries_1D):

    dct = {}
    def make(*keys):
        dct[keys] = lib.make_symcon(fixt_symmetries_1D, *keys)

    make("ijk,++-",       "klm,++-",       "ijlm,+++-")
    make("inom,+++-",     "inp,++-",       "pom,--+")
    make("ixnyom,+0+0+-", "inzxp,++00-",   "pyzom,-00-+")

    make("ijk,++-,Q1,M3", "klm,++-,Q2,M3", "ijlm,+++-,Q3,M3")
    make("ijk,++-,Q1,M3", "klm,-+-,Q2,M3", "ijlm,--+-,Q1,M3")
    make("ijk,++-,M3",    "klm,++-",       "ijlm,+++-,M3")
    make("ijk,++-",       "klm,++-,M3",    "ijlm,+++-,M3")
    make("ijk,++-,M2",    "klm,++-",       "ijlm,+++-,M2")
    make("ijk,++-",       "klm,++-,M5",    "ijlm,+++-,M5")

    make("ijlm,+++-",     "nojl,++--",     "inom,+++-")
    make("ijlm,+++-,M2",  "nojl,++--",     "inom,+++-,M2")
    make("ijlm,+++-",     "nojl,++--,M2",  "inom,+++-,M2")

    return dct




# --- Symmetries and symmetry contractions, 3D ------------------------------ #


@pytest.fixture
def fixt_mod_3D():
    mod  = (2 * np.pi / 5) * np.eye(3)
    return mod




@pytest.fixture
def fixt_symlabels_3D(fixt_mod_3D):

    dct = {}
    def make(mod, key):
        dct[key] = lib.make_symlabels_3D(mod, key)

    make(fixt_mod_3D, (3,3,1))
    make(fixt_mod_3D, (2,2,1))

    return dct




@pytest.fixture
def fixt_symmetries_3D(fixt_symlabels_3D, fixt_mod_3D):

    kpts = fixt_symlabels_3D[(3,3,1)]
    mod  = fixt_mod_3D 
 
    dct = {}
    def make(legs, fullsigns, suffix=None, **kwargs):

        key, out = lib.make_symmetry_3D_wrap(legs, fullsigns, \
                                             kpts, suffix, **kwargs)
        dct[key] = out

    make("ijlm", "+++-", suffix="M1", mod=mod)
    make("nojl", "++--", suffix="M1", mod=mod)
    make("ijk",  "++-",  suffix="M1", mod=mod)
    make("klm",  "++-",  suffix="M1", mod=mod)
    make("klm",  "++-",  suffix="M2", mod=2*mod)

    make("inom", "+++-",  suffix="M1", mod=mod)
    make("inp",   "++-",  suffix="M1", mod=mod)
    make("pom",   "--+",  suffix="M1", mod=mod)

    make("ixnyom", "+0+0+-", suffix="M1", mod=mod, signs="+++-")
    make("inzxp",  "++00-",  suffix="M1", mod=mod, signs="++-")
    make("pyzom",  "-00-+",  suffix="M1", mod=mod, signs="--+")

    sym = tn.Symmetry1D("++-", [np.arange(0,2)]*3)
    dct["klm,++-,Invalid,1D"] = (sym, [], Str("klm"), Str("KLM"))

    return dct




@pytest.fixture
def fixt_symcon_3D(fixt_symmetries_3D):

    dct = {}
    def make(*keys):
        dct[keys] = lib.make_symcon(fixt_symmetries_3D, *keys)

    make("ijk,++-,M1",       "klm,++-,M1",     "ijlm,+++-,M1")
    make("ijlm,+++-,M1",     "nojl,++--,M1",   "inom,+++-,M1")
    make("inom,+++-,M1",     "inp,++-,M1",     "pom,--+,M1")
    make("ixnyom,+0+0+-,M1", "inzxp,++00-,M1", "pyzom,-00-+,M1")

    return dct




@pytest.fixture
def fixt_symmetries_3D_1(fixt_symlabels_3D, fixt_mod_3D):

    kpts = fixt_symlabels_3D[(2,2,1)]
    mod  = fixt_mod_3D
    
    dct = {}
    def make(legs, fullsigns, suffix=None, **kwargs):

        key, out = lib.make_symmetry_3D_wrap(legs, fullsigns, \
                                             kpts, suffix, **kwargs)
        dct[key] = out

    make("ij",   "+-",   suffix="M1",    mod=mod)
    make("ijkl", "+-+-", suffix="M1",    mod=mod)
    make("ijkl", "+-+-", suffix="M1",    mod=mod)
    make("ijkl", "++--", suffix="M1",    mod=mod)
    make("ijk",  "++-",  suffix="Q1,M1", mod=mod, qtot=kpts[2])
    make("i",    "+",    suffix="Q1,M1", mod=mod, qtot=kpts[2])

    return dct




# --- Auxiliary quantities for symmetries ----------------------------------- #

@pytest.fixture
def fixt_aux_symlabels(fixt_symcon_1D):   

    dct = {}
    def make(*key):

        symcon, sym, _, _ = fixt_symcon_1D[key]

        phase = -symcon.phase
        symA  = sym["A"]
        symB  = sym["B"]

        def _make(symindicesA, symindicesB):
              
            Saux_A = lib.make_aux_symlabels(symA, symindicesA)
            Saux_B = lib.make_aux_symlabels(symB, symindicesB, phase=phase)
            Saux   = lib.align_symlabels(Saux_A, Saux_B)

            dct[key] = Saux

        return _make

    symidxA = [(0,1), (2,3)]
    symidxB = [(0,1), (2,)]

    make("inom,+++-",     "inp,++-",      "pom,--+"    )(symidxA, symidxB)
    make("ixnyom,+0+0+-", "inzxp,++00-",  "pyzom,-00-+")(symidxA, symidxB)

    symidxA = [(1,2), (0,3)]
    symidxB = [(2,3), (0,1)]

    make("ijlm,+++-",      "nojl,++--",    "inom,+++-"   )(symidxA, symidxB)
    make("ijlm,+++-,M2",   "nojl,++--",    "inom,+++-,M2")(symidxA, symidxB)
    make("ijlm,+++-",      "nojl,++--,M2", "inom,+++-,M2")(symidxA, symidxB)

    return dct




@pytest.fixture
def fixt_aligned_symlabels():  

    dct = {}
    def make(idx, *key):

        def _make(A, B, ans, ans1):
            dct[(idx, *key)] = (A, B, ans, ans1)

        return _make

    A = np.arange(0,5)
    B = np.arange(0,2)
    C = np.arange(1,5)

    ans  = np.array([0,1])
    ans1 = lib.align_symlabels(A, B)
    make(1, "ijk,++-",  "klm,++-",  "ijlm,+++-")(A, B, ans, ans1)

    ans  = np.array([1,2,3,4])
    ans1 = lib.align_symlabels(A, C)
    make(1, "ijk,++-,Q1,M3", "klm,++-,Q2,M3", \
                             "ijlm,+++-,Q3,M3")(A, C, ans, ans1)

    ans  = np.array([1,2])
    ans1 = lib.align_symlabels(C, lib.fold_1D(A, 3))
    make(1, "ijk,++-",    "klm,++-,M3", "ijlm,+++-,M3")(A, C, ans, ans1)

    ans  = np.array([0,1,2])
    ans1 = lib.align_symlabels(A, lib.fold_1D(C, 3))
    make(1, "ijk,++-,M3", "klm,++-",    "ijlm,+++-,M3")(A, C, ans, ans1)

    return dct




# --- Maps ------------------------------------------------------------------ #

@pytest.fixture
def fixt_maps(fixt_symcon_1D, fixt_symlabels_1D, fixt_aux_symlabels):

    key       = ("ijlm,+++-", "nojl,++--", "inom,+++-")
    symlabels = fixt_symlabels_1D

    # Get symmetries
    symcon, sym, legs, symlegs = fixt_symcon_1D[key]

    Saux      = fixt_aux_symlabels[key]
    symlabels = [symlabels["j"], symlabels["l"], Saux]
    aux_sym   = tn.Symmetry1D("++-", symlabels)

    # Compute initial maps
    mapA = tn.Map.compute(sym["A"], Str("IJLM")) 
    mapB = tn.Map.compute(sym["B"], Str("NOJL"))
    mapQ = tn.Map.compute(aux_sym,  Str("JLQ"))
  
    # Compute map contractions
    maps, legs, shapes = lib.make_map_contractions(mapA, mapB, mapQ, Saux)

    # Get leg dims
    legdims      = {l.upper(): len(s) for l,s in fixt_symlabels_1D.items()}
    legdims["Q"] = len(Saux)

    return key, maps, legs, legdims, shapes




@pytest.fixture
def fixt_random_maps(fixt_aux_symlabels):

    # Auxiliary symlabels (for shape)
    Saux = fixt_aux_symlabels[("ijlm,+++-", "nojl,++--", "inom,+++-")]

    # Compute initial random maps
    mapA, _ = lib.make_random_map(Str("IJLM"), (5,2,4,4),       14)
    mapB, _ = lib.make_random_map(Str("NOJL"), (5,2,2,4),       10)
    mapQ, _ = lib.make_random_map(Str("JLQ"),  (2,4,len(Saux)), 10)

    initial_maps = [mapA, mapB, mapQ]

    # Compute map contractions
    maps, legs, shapes = lib.make_map_contractions(mapA, mapB, mapQ, Saux)

    return initial_maps, maps, legs, shapes




# --- Transformations ------------------------------------------------------- #

@pytest.fixture
def fixt_transform_data(fixt_symcon_1D, fixt_maps):

    trans_data = lib.TransformData(fixt_symcon_1D, fixt_maps)
    return trans_data




# --- Tensors, with 1D and 3D symmetries ------------------------------------------------------- #

@pytest.fixture
def fixt_tensors_1D(fixt_symmetries_1D):

    dct = {}
    def make(dense_shape, symkey=None):

        key, out = lib.make_tensor_wrap(dense_shape, fixt_symmetries_1D, symkey)
        dct[key] = out

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




@pytest.fixture
def fixt_tensors_3D(fixt_symmetries_3D_1):

    dct = {}
    def make(dense_shape, symkey=None):

        key, out = lib.make_tensor_wrap(dense_shape, \
                                        fixt_symmetries_3D_1, symkey)
        dct[key] = out

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

























































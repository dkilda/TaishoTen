#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import copy  as cp
import numpy as np
import helper_lib as lib

import itertools

import util
from util import isiterable, noniterable

import taishoten as tn
from taishoten import Str
from taishoten.util import IS, ISNOT, ARE, ARENOT



# --- Test Str class -------------------------------------------------------- #

class TestStr:

   @pytest.mark.parametrize("x, ans", \
   [                                  \
   ["IJKL",                None],     \
   [{"I", "J", "K", "L"},  "IJKL"],   \
   [["I", "J", "K", "L"],  "IJKL"],   \
   ["JLIK",                None],     \
   ["mnik",                None],     \
   ["IJKdabc",             None],     \
   ])
   def test_construct(self, x, ans):

       if  ans is None:
           ans = x

       s = Str(x)
       util.assert_Str(s, ans)



   @pytest.mark.parametrize("x", ["IJKL"])
   def test_iter(self, x):

       s1 = Str(x)

       # Iterator test
       it1 = iter(s1)
       out = []
       while True:
           try:
              v = next(it1) 
              out.append(v)
           except StopIteration:
              break
   
       out = Str(out)
       assert out == Str(x)



   @pytest.mark.parametrize("x, ans", \
   [                                  \
   ["IJKL", [(Str("I"), Str("J")),    \
             (Str("I"), Str("K")),    \
             (Str("I"), Str("L")),    \
             (Str("J"), Str("K")),    \
             (Str("J"), Str("L")),    \
             (Str("K"), Str("L"))]],  \
   ])
   def test_itertools(self, x, ans):

       s1  = Str(x)
       out = [cmb for cmb in itertools.combinations(s1, 2)]

       assert len(out) == len(ans)
       assert all(v in ans for v in out)



   @pytest.mark.parametrize("x, y", [["IJKL", "MNOP"]])
   def test_hash(self, x, y):

       s1 = Str(x)
       s2 = Str(x)
       s3 = Str(y)

       assert hash(s1) == hash(s2)
       assert hash(s1) != hash(s3)



   @pytest.mark.parametrize("x, y, z", [["IJKL", "DJKL", "NJKL"], \
                                        ["IJKL", "IJK",  "ijkl"], \
                                        ["IJKL", "AEW",  "ijkl"], \
                                        ["IJKL", "AEW",  "abce"], \
                                        ["mijn", "ABCE", "nojl"]])
   def test_comparisons(self, x, y, z):

       s1 = Str(x)
       s2 = Str(x)
       s3 = Str(y)
       s4 = Str(z)

       assert s1 == s2
       assert s1 != s3
       assert s1 != s4

       assert s1 > s3
       assert s3 < s1

       assert s1 < s4
       assert s4 > s1

       assert s1 >= s3
       assert s1 <= s4

       assert s1 >= s2
       assert s1 <= s2



   @pytest.mark.parametrize("x, y, ans",     \
   [                                         \
   ["ijlm",     "nojl",     "ijlmno"],       \
   ["nojl",     "ijlm",     "nojlim"],       \
   ["damc",     "IJap",     "damcIJp"],      \
   ["ijcxdIJC", "abcxdABC", "ijcxdIJCabAB"], \
   ])
   def test_add(self, x, y, ans):

       x   = Str(x)
       y   = Str(y)
       ans = Str(ans)

       out = x + y
       assert out == ans

       
       
   @pytest.mark.parametrize("x, y, ans", \
   [                                     \
   ["ijlm",     "nojl",     "im"],       \
   ["nojl",     "ijlm",     "no"],       \
   ["damc",     "IJap",     "cdm"],      \
   ["aJIp",     "dpmc",     "IJa"],      \
   ["cabMKL",   "Lobl",     "KMac"],     \
   ["ijcxdIJC", "abcxdABC", "IJij"],     \
   ["abcxdABC", "ijcxdIJC", "ABab"],     \
   ])
   def test_sub(self, x, y, ans):

       x   = Str(x)
       y   = Str(y)
       ans = Str(ans)

       out = x - y
       assert out == ans



   @pytest.mark.parametrize("x, y, ans", \
   [                                     \
   ["ijlm",     "nojl",     "jl"],       \
   ["nojlNLJ",  "ijlmIJL",  "JLjl"],     \
   ["ijcxdIJC", "abcxdABC", "Ccdx"],   \
   ])
   def test_and(self, x, y, ans):

       x   = Str(x)
       y   = Str(y)
       ans = Str(ans)

       out = x & y
       assert out == ans



   @pytest.mark.parametrize("x, y, ans",     \
   [                                         \
   ["ijlm",     "nojl",     "ijlmno"],       \
   ["nojlNLJ",  "ijlmIJL",  "IJLNijlmno"],   \
   ["ijcxdIJC", "abcxdABC", "ABCIJabcdijx"], \
   ])
   def test_or(self, x, y, ans):

       x   = Str(x)
       y   = Str(y)
       ans = Str(ans)

       out = x | y
       assert out == ans



   @pytest.mark.parametrize("x, y, ans",     \
   [                                         \
   ["ijlm",     "nojl",     "imno"],         \
   ["nojlNLJ",  "ijlmIJL",  "INimno"],       \
   ["ijcxdIJC", "abcxdABC", "ABIJabij"],     \
   ])
   def test_xor(self, x, y, ans):

       x   = Str(x)
       y   = Str(y)
       ans = Str(ans)

       out = x ^ y
       assert out == ans



   @pytest.mark.parametrize("x, y, ans", \
   [                                     \
   ["IJK",  "IJKL",  True],              \
   ["JIK",  "IJKL",  True],              \
   ["MJK",  "MJAB",  False],             \
   ["IJKL", "IJK",   False],             \
   ])
   def test_issubset(self, x, y, ans):

       x   = Str(x)
       y   = Str(y)

       out = x.issubset(y)
       assert out == ans   



   @pytest.mark.parametrize("x, y, ans", \
   [                                     \
   ["IJK",  "IJKL",  True],              \
   ["JIK",  "IJKL",  True],              \
   ["MJK",  "MJAB",  False],             \
   ["IJKL", "IJK",   False],             \
   ])
   def test_in(self, x, y, ans):

       x   = Str(x)
       y   = Str(y)

       out = (x in y)
       assert out == ans   



   @pytest.mark.parametrize("xs, ans",                       \
   [                                                         \
   [("ijlm",     "nojl"),                "ijlmno"],          \
   [("nojl",     "ijlm"),                "nojlim"],          \
   [("damc",     "IJap"),                "damcIJp"],         \
   [("ijcxdIJC", "abcxdABC"),            "ijcxdIJCabAB"],    \
   [("ijlm", "nojl", "IJap"),            "ijlmnoIJap"],      \
   [("ijlm", "nojl", "IJap", "KROkinw"), "ijlmnoIJapKROkw"], \
   ])
   def test_join(self, xs, ans):

       xs  = tuple([Str(x) for x in xs])
       ans = Str(ans)

       out = tn.util.join(*xs)

       assert out      == ans
       assert len(out) == len(ans)




# --- Test functions in util module ----------------------------------------- #


@pytest.fixture
def leg_triplet_fixt():

    dct = {}

    # Entry-1, "IJLM,NOJL->INOM"
    legs  = {"A": Str("IJLM"), "B": Str("NOJL"), "C": Str("INOM")}
    items = {"A": [1,2,3,4],   "B": [5,6,21,31], "C": [1,5,6,4]  }

    ans_compress   = [1,5,6,4]
    ans_shared     = [(2,21), (3,31)]
    ans_shared_idx = [(1,2),  (2,3)]

    dct["IJLM,NOJL->INOM"] = (legs, items, ans_compress, \
                              ans_shared, ans_shared_idx)

    # Entry-2, "IxzJLM,NOyJxL->INOMyz"
    legs  = {"A": Str("IxzJLM"), \
             "B": Str("NOyJxL"), \
             "C": Str("INOMyz")  }

    items = {"A": [1, '*',  '#',   2,     3,    4], \
             "B": [5,   6,  '!',  21,  '**',   31], \
             "C": [1,   5,    6,   4,   '!',  '#']  }

    ans_compress   = [1,5,6,4,'!','#']
    ans_shared     = [(2,21), (3,31), ('*','**')]
    ans_shared_idx = [(3,4,1), (3,5,4)]

    dct["IxzJLM,NOyJxL->INOMyz"] = (legs, items, ans_compress, \
                                    ans_shared, ans_shared_idx)
    return dct




@pytest.fixture
def legs_list_fixt():

    # Dummy object with legs
    class Box:

       def __init__(self, a, legs):
           self.a    = a
           self.legs = legs

       def __eq__(self, other):
           return (self.a == other.a) and (self.legs == other.legs)


    # Make a list of legs and boxes
    legs_lst = [Str("molw"),    Str("IJK"), Str("IMNO"), \
                Str("POMpoml"), Str("PQ"),  Str("PQL"), Str("ABC")]

    boxes = [Box(i, legs) for i, legs in enumerate(legs_lst)]


    # Make a sorted list of legs and boxes    
    sorted_legs_lst = [Str("PQ"),  Str("ABC"),  Str("IJK"), \
                       Str("PQL"), Str("IMNO"), Str("molw"), Str("POMpoml")]

    sorted_boxes = [boxes[i] for i in (4,6,1,5,2,0,3)]

    return boxes, legs_lst, sorted_boxes, sorted_legs_lst





class TestUtil:

   # --- Test booleans and assertions --------------------------------------- #

   @pytest.mark.parametrize("x, ans", \
   [                \
    [ 0,    True],  \
    [ 1,    True],  \
    [-1,    True],  \
    ["IJK", True],  \
    [None,  False], \
   ])
   def test_IS(self, x, ans):

       assert IS(x)    == ans
       assert ISNOT(x) == (not ans)



   @pytest.mark.parametrize("x, ans", \
   [                                  \
    [[0, 1, -1,   "IJK"],  True],     \
    [[0, 1, "IJK", None],  False],    \
    [[0, 1,  None],        False],    \
   ])
   def test_ARE(self, x, ans):

       assert ARE(*x) == ans



   @pytest.mark.parametrize("x, ans", \
   [                                  \
    [[0, 1, -1,   "IJK"],  False],    \
    [[0, 1, "IJK", None],  False],    \
    [[None, None, None],   True],    \
   ])
   def test_ARENOT(self, x, ans):

       assert ARENOT(*x) == ans



   @pytest.mark.parametrize("x, y", \
   [                                \
    [1, 1],                         \
    ["IJK", "IJK"],                 \
    [(2,3,5), (2,3,5)],             \
   ])
   def test_assertequal(self, x, y):
       tn.util.assertequal(x, y, "")



   @pytest.mark.parametrize("x, y", \
   [                                \
    [1, 2],                         \
    ["IJK", "ILJK"],                 \
    [(2,3,5), (2,4,5)],             \
   ])
   @pytest.mark.xfail
   def test_assertequal_failed(self, x, y):
       tn.util.assertequal(x, y, "")



   @pytest.mark.parametrize("shape", [(3,4,5)])
   def test_assertclose(self, shape):
 
       np.random.seed(1)
       x = lib.randn(*shape)
       y = x.copy()

       tn.util.assertclose(x, y, "")



   @pytest.mark.parametrize("shape, shape1", [[(3,4,5), None],    \
                                              [(3,4,5), (3,4,6)], \
                                             ])
   @pytest.mark.xfail
   def test_assertclose_failed(self, shape, shape1):
 
       if shape1 is None:
          shape1 = shape

       np.random.seed(1)
       x = lib.rand(shape)
       y = lib.rand(shape1)

       tn.util.assertclose(x, y, "")



   @pytest.mark.parametrize("x, y", \
   [                                \
    [2, [1,2,5,6,9]],               \
    [9, [1,2,5,6,9]],               \
    ["IJK", "IJKL"],                \
   ])
   def test_assertin(self, x, y):
       tn.util.assertin(x, y, "")



   @pytest.mark.parametrize("x, y", \
   [                                \
    [7,  [1,2,5,6,9]],              \
    [-1, [1,2,5,6,9]],              \
    ["IJM", "IJKL"],                \
   ])
   @pytest.mark.xfail
   def test_assertin_failed(self, x, y):
       tn.util.assertin(x, y, "")



   @pytest.mark.parametrize("x", [[1,2,3,4,5]])
   def test_assertunique(self, x):
       tn.util.assertunique(x, "")



   @pytest.mark.parametrize("x", [[1,2,3,2,5]])
   @pytest.mark.xfail
   def test_assertunique_failed(self, x):
       tn.util.assertunique(x, "")



   @pytest.mark.parametrize("x, ans", \
   [                 \
   [ 3,      False], \
   [ 1.2,    False], \
   [[1],     True],  \
   [[1,2,3], True],  \
   ["IJKL",  True],  \
   ])
   def test_isiterable(self, x, ans):

       assert tn.util.isiterable(x)  == ans
       assert tn.util.noniterable(x) == (not ans)



   # --- Test itertools wrappers -------------------------------------------- #

   @pytest.mark.parametrize("xs", [[Str("IJK"), Str("ABC"), Str("XYZW")]])
   def test_itertools_cartesian_prod(self, xs):

       # Test Cartesian product
       out = list(tn.util.cartesian_prod(*xs))
       ans = list(itertools.product(*xs))     

       util.assert_list(out, ans)



   @pytest.mark.parametrize("xs, keys", \
   [[[Str("IJK"), Str("ABC"), Str("XYZW")], ["A", "B", "C"]]])
   def test_itertools_cartesian_prod_dict(self, xs, keys):

       # Test dictionary Cartesian product
       dct = dict(zip(keys, xs))

       out = list(tn.util.cartesian_prod_dict(dct, keys))
       ans = list(itertools.product(*xs))     

       util.assert_list(out, ans)



   @pytest.mark.parametrize("xs, r", [[Str("XYZW"), 2]])
   def test_itertools_combinations(self, xs, r):

       # Test combinations
       out = list(tn.util.combinations(xs, r))
       ans = list(itertools.combinations(xs, r))     

       util.assert_list(out, ans)



   @pytest.mark.parametrize("xs, r", [[Str("XYZW"), 2]])
   def test_itertools_idx_combinations(self, xs, r):

       # Test index combinations
       out = list(tn.util.combinations(xs, r))

       ans_idx = list(itertools.combinations(range(len(xs)), r)) 
       ans     = [(xs[i], xs[j]) for i,j in ans_idx]    

       util.assert_list(out, ans)



   # --- Test signs --------------------------------------------------------- #

   @pytest.mark.parametrize("x, ans", [["++--",  [1,1,-1,-1]],   \
                                       ["-+-++", [-1,1,-1,1,1]], \
                                      ])
   def test_signs_to_int(self, x, ans):
       assert tn.util.signs_to_int(x) == ans


  
   @pytest.mark.parametrize("x, ans", [["++--",     "--++"],    \
                                       ["-+-++",    "+-+--"],   \
                                       ["-+-0++0",  "+-+0--0"], \
                                      ])
   def test_flip_signs(self, x, ans):
       assert tn.util.flip_signs(x) == ans



   @pytest.mark.parametrize("x, ans", [["++--",     "--++"],    \
                                       ["-+-++",    "+-+--"],   \
                                       ["-+-0++0",  "+-+0--0"], \
                                      ])
   def test_phased_signs(self, x, ans):
       assert tn.util.phased_signs(x,  1) == x
       assert tn.util.phased_signs(x, -1) == ans



   # --- Test legs ---------------------------------------------------------- #

   @pytest.mark.parametrize("x, ans",        \
   [                                         \
   [[Str("IJLM"), Str("NOJL"), Str("INOM")], \
    [Str("IJL"),  Str("NOJ"),  Str("INO")]], \

   [Str("ijlm"),  Str("ijl")],               \
   ])
   def test_truncate(self, x, ans):
       assert tn.util.truncate(x) == ans



   @pytest.mark.parametrize("x, signs, ans",  \
   [                                          \
    [Str("ijlm"),    "-++-",    Str("ijlm")], \
    [Str("ijlxm"),   "-++0-",   Str("ijlm")], \
    [Str("iyzjlxm"), "-00++0-", Str("ijlm")], \
   ])
   def test_cut_unsigned(self, x, signs, ans): 
       assert tn.util.cut_unsigned(x, signs) == ans



   @pytest.mark.parametrize("x, signs, ans",  \
   [                                          \
    [Str("ijlm"),    "-++-",    Str("IJLM")], \
    [Str("ijlxm"),   "-++0-",   Str("IJLM")], \
    [Str("iyzjlxm"), "-00++0-", Str("IJLM")], \
   ])
   def test_make_symlegs(self, x, signs, ans): 
       assert tn.util.make_symlegs(x, signs) == ans



   @pytest.mark.parametrize("num_legs, ans",  \
   [                                          \
    [5,  Str("abcde")],                       \
    [21, Str("abcdefghijklmnoprstuv")],       \
   ])
   def test_make_legs(self, num_legs, ans): 

       out = tn.util.make_legs(num_legs)
       assert out == ans
       assert Str("q") not in out



   @pytest.mark.parametrize("x, ans", \
   [[[Str("IJLM"), Str("NOJL"), Str("INOM")], 6]])
   def test_get_num_legs(self, x, ans): 

       out = tn.util.get_num_legs(*x)
       assert out == ans



   @pytest.mark.parametrize("x, ans",                            \
   [                                                             \
   ["IJLM,NOJL->INOM", [Str("IJLM"), Str("NOJL"), Str("INOM")]], \
   ["IJLM,IJLM->",     [Str("IJLM"), Str("IJLM"), Str("")    ]], \
   ])
   def test_subscript_to_legs(self, x, ans):

       out = tn.util.subscript_to_legs(x)
       assert out == ans



   @pytest.mark.parametrize("x, ans",                            \
   [                                                             \
   [[Str("IJLM"), Str("NOJL"), Str("INOM")], "IJLM,NOJL->INOM"], \
   [[Str("IJLM"), Str("NOJL"), Str("")],     "IJLM,NOJL->"    ], \
   ])
   def test_legs_to_subscript(self, x, ans):

       out = tn.util.legs_to_subscript(*x)
       assert out == ans



   # --- Test generators ---------------------------------------------------- #

   @pytest.mark.parametrize("key", \
                           ["IJLM,NOJL->INOM", "IxzJLM,NOyJxL->INOMyz"])                                          
   def test_zip_compress(self, leg_triplet_fixt, key):

       legs, items, ans, _, _ = leg_triplet_fixt[key]

       a = (legs["A"], items["A"])
       b = (legs["B"], items["B"])

       out = tn.util.zip_compress(a, b, legs["C"])
       out = list(out)
       
       util.assert_list(out, ans)

       
   @pytest.mark.parametrize("key", \
                           ["IJLM,NOJL->INOM", "IxzJLM,NOyJxL->INOMyz"])     
   def test_zip_shared(self, leg_triplet_fixt, key):

       legs, items, _, ans, _ = leg_triplet_fixt[key]

       a = (legs["A"], items["A"])
       b = (legs["B"], items["B"])

       out = tn.util.zip_shared(a, b)
       out = list(out)
       
       util.assert_list(out, ans)



   @pytest.mark.parametrize("key", \
                           ["IJLM,NOJL->INOM", "IxzJLM,NOyJxL->INOMyz"])     
   def test_get_shared_indices(self, leg_triplet_fixt, key):

       legs, _, _, _, ans = leg_triplet_fixt[key]

       out = tn.util.get_shared_indices(legs["A"], legs["B"])
       out = list(out)
       
       util.assert_list(out, ans)



   # --- Test iterables ----------------------------------------------------- #

   @pytest.mark.parametrize("keys, vals, ans",  \
   [[                                           \
    ("A", "B", "C"),                            \
    ["1234", "5623", "1564"],                   \
    {"A": "1234", "B": "5623", "C": "1564"},    \
   ]])  
   def test_dictzip(self, keys, vals, ans):

       out = tn.util.dictzip(keys, vals)
       util.assert_dict(out, ans)



   @pytest.mark.parametrize("x, ans",         \
   [[                                         \
    ["1234", "5623", "1564"],                 \
    {"A": "1234", "B": "5623", "C": "1564"},  \
   ]])  
   def test_dictriplet(self, x, ans):

       out = tn.util.dictriplet(*x)
       util.assert_dict(out, ans)



   @pytest.mark.parametrize("lst, idx, ans",  \
   [[                                         \
    ["A","B","C","D","E","F","G"],            \
    (1,2,4,6),                                \
    ["B","C","E","G"],                        \
   ]])  
   def test_get_items(self, lst, idx, ans):

       out = tn.util.get_items(lst, idx)
       util.assert_list(out, ans)



   @pytest.mark.parametrize("x, ans", [[range(5), "01234"]])  
   def test_to_string(self, x, ans):

       out = tn.util.to_string(x)
       assert out == ans



   @pytest.mark.parametrize("x, ans",  \
   [                                   \
    ["MINKJ",     "IJKMN"],            \
    ["cadbMINKJ", "IJKMNabcd"],        \
   ])  
   def test_sorted_string(self, x, ans):

       out = tn.util.sorted_string(x)
       assert out == ans



   def test_sort(self, legs_list_fixt):

       _, legs, _, sorted_legs = legs_list_fixt

       out = tn.util.sort(legs)
       assert out == sorted_legs



   def test_sort_by_legs(self, legs_list_fixt):

       boxes, _, sorted_boxes, _ = legs_list_fixt

       out = tn.util.sort_by_legs(boxes)
       assert out == sorted_boxes



   def test_get_legs(self, legs_list_fixt):

       boxes, legs, _, _ = legs_list_fixt

       out = tn.util.get_legs(boxes)
       assert out == legs



   @pytest.mark.parametrize("legs, ans_idx",                \
   [                                                        \
   [Str("IMNO"), 2], [Str("molw"), 0], [Str("POMpoml"), 3], \
   ]) 
   def test_get_from_legs(self, legs_list_fixt, legs, ans_idx):

       boxes, _, _, _ = legs_list_fixt

       out = tn.util.get_from_legs(boxes, legs)
       assert out == boxes[ans_idx]
































































































































































































































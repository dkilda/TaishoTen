#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import itertools

import numpy      as np
import helper_lib as lib
import taishoten  as tn

from taishoten import Str
from taishoten import IS, ISNOT, ARE, ARENOT
from .util     import TaishoTenTestCase, must_fail



# --- Test Str class -------------------------------------------------------- #

class TestStr(TaishoTenTestCase):

   def test_construct(self):

       s1 = Str("IJKL")
       self.assertStr(s1, "IJKL")

       s2 = Str({"I", "J", "K", "L"})
       self.assertStr(s2, "IJKL")

       s3 = Str(["I", "J", "K", "L"])
       self.assertStr(s3, "IJKL")

       s4 = Str("JLIK")
       self.assertStr(s4, "JLIK")

       s5 = Str("mnik")
       self.assertStr(s5, "mnik")

       s6 = Str("IJKlijk")
       self.assertStr(s6, "IJKlijk")


   def test_iter(self):

       s1 = Str("IJKL")

       # Iterator test
       it1 = iter(s1)
       ans = []
       while True:
           try:
              v = next(it1) 
              ans.append(v)
           except StopIteration:
              break
   
       ans = Str(ans)
       assert ans == Str("IJKL")

       # Itertools test
       ans = [Str("IJ"), Str("IK"), Str("IL"), \
              Str("JK"), Str("JL"), Str("KL")]

       combos = [cmb for cmb in itertools.combinations(s1, 2)]

       assert len(ans) == len(combos)
       for cmb in combos:
           assert cmb in ans


   def test_hash(self):

       s1 = Str("IJKL")
       s2 = Str("IJKL")
       s3 = Str("MNOP")

       assert hash(s1) == hash(s2)
       assert hash(s1) != hash(s3)


   def test_comparisons(self):

       s1 = Str("IJKL")
       s2 = Str("IJKL")
       s3 = Str("JILK")
       s4 = Str("IJK")
       s5 = Str("JIL")
       s6 = Str("MIJN")
       s7 = Str("ijkl")

       # Test-1
       assert s1 == s2
       assert s1 != s3

       # Test-2
       assert s3 >  s1
       assert s1 <  s3

       # Test-3
       assert s4 <  s1
       assert s1 >  s4

       # Test-4
       assert s5 >  s1
       assert s1 <  s5

       # Test-5
       assert s6 >= s1
       assert s1 <= s6

       # Test-6
       assert s1 < s7
       assert s7 > s1       
              

   def test_algebra(self):

       sA = Str("IJLM")
       sB = Str("NOJL")
       sC = Str("INOM")

       out = sA + sB
       assert out == Str("IJLMNO")

       out = sB + sA
       assert out == Str("IJLMNO")

       out = sB + sC
       assert out == Str("IJLMNO")

       out = sA - sB
       assert out == Str("IM")

       out = sA | sB
       assert out == Str("IJLMNO")

       out = sA & sB
       assert out == Str("JL")

       out = sA ^ sB
       assert out == Str("IMNO")

       out = sB ^ sC
       assert out == Str("IJLM")

       assert not sA.issubset(sB)
       assert     sA.issubset(Str("ABIJLM"))

       out = tn.join(sA, sB, sC)
       assert out      == Str("IJLMNO")
       assert len(out) == 6




# --- Test functions in util module ----------------------------------------- #

class TestUtil(TaishoTenTestCase):

   def test_booleans(self):

       a = 0
       b = 1
       c = None
       d = None

       # Test IS/ISNOT
       assert IS(a)
       assert IS(b)
       assert ISNOT(c)
       assert ISNOT(d)

       assert not IS(c)
       assert not IS(d)
       assert not ISNOT(a)
       assert not ISNOT(b)

       # Test ARE/ARENOT
       assert ARE(a,b)
       assert ARENOT(c,d)
       assert ARENOT(a,b,c)
 
       assert not ARE(c,d)
       assert not ARE(a,b,c)
       assert not ARENOT(a,b)

       assert ARE(a)
       assert ARENOT(c)

       

   def test_assertions(self):

       # Test assertequal
       a = 1
       b = 1
       c = 2

       tn.assertequal(a, b, "test_assertions")
       must_fail(tn.assertequal)(a, c, "test_assertions")
       must_fail(tn.assertequal)(b, c, "test_assertions")

       # Test assertclose
       a = lib.randn(3,4,5)
       b = a.copy()
       c = lib.randn(3,4,5)
       d = lib.randn(3,4,5,6)

       tn.assertclose(a, b, "test_assertions")
       must_fail(tn.assertequal)(a, c, "test_assertions")
       must_fail(tn.assertequal)(a, d, "test_assertions")
       
       # Test assertin  
       vals = [1,2,5,6,9]

       tn.assertin(2, vals)
       tn.assertin(9, vals)
       must_fail(tn.assertin)(3, vals, "test_assertions")
       must_fail(tn.assertin)(7, vals, "test_assertions")
   
       # Test assertunique
       vals1 = [1,1,1,1,1]
       vals2 = [1,1,2,1,1]

       tn.assertunique(vals1,"test_assertions")
       must_fail(tn.assertunique)(vals2, "test_assertions")
       
       # Test isiterable/noniterable 
       a = [1]
       b = [1,2,3]
       c = 3

       assert tn.isiterable(a)
       assert tn.isiterable(b)
       assert tn.noniterable(c)

       must_fail(tn.isiterable)(c,  "test_assertions")
       must_fail(tn.noniterable)(a, "test_assertions")
       must_fail(tn.noniterable)(b, "test_assertions")



   def test_itertools_wrappers(self):

       a = Str("IJK")
       b = Str("ABC")
       c = Str("XYZW")

       # Test Cartesian product
       out = list(tn.cartesian_prod(a,b,c))
       ans = list(itertools.product(a,b,c))

       util.assertList(out, ans)

       # Test dictionary Cartesian product 
       dct = {"A": a, "B": b, "C": c}
       out = list(tn.cartesian_prod_dict(dct, ["A", "B", "C"]))
       ans = list(itertools.product(a,b,c))

       util.assertList(out, ans)

       # Test combinations
       out = list(tn.combinations(c, 2))
       ans = list(itertools.combinations(c, 2))

       util.assertList(out, ans)

       # Test index combinations
       out = list(tn.idx_combinations(c, 2))
       ans = list(itertools.combinations(len(c), 2))
       
       util.assertList(out, ans)



   def test_subscript(self):

       # Test subscript_to_legs
       sub  = "IJLM,NOJL->INOM"
       legs = [Str("IJLM"), Str("NOJL"), Str("INOM")]

       out = tn.subscript_to_legs(sub)
       assert out == legs

       # Test legs_to_subscript
       out = tn.legs_to_subscript(legs)
       assert out == sub



   def test_signs(self):

       signs1 = "++--"
       signs2 = "-+-++"
       signs3 = "-+-0++0"

       # Test signs_to_int
       assert tn.signs_to_ints(signs1) == (1,1,-1,-1)
       assert tn.signs_to_ints(signs2) == (-1,1,-1,1,1)

       # Test flip_signs
       assert tn.flip_signs(signs1) == "--++"
       assert tn.flip_signs(signs2) == "+-+--"
       assert tn.flip_signs(signs3) == "+-+0--0"

       # Test phased_signs
       assert tn.phased_signs(signs1) == signs1
       assert tn.phased_signs(signs2) == signs2
       assert tn.phased_signs(signs3) == signs3

       assert tn.phased_signs(signs1, 1) == signs1
       assert tn.phased_signs(signs2, 1) == signs2
       assert tn.phased_signs(signs3, 1) == signs3

       assert tn.phased_signs(signs1, -1) == "--++"
       assert tn.phased_signs(signs2, -1) == "+-+--"
       assert tn.phased_signs(signs3, -1) == "+-+0--0"



   def test_legs(self):

       legs1 = [Str("IJLM"), Str("NOJL"), Str("INOM")]
       legs2 = Str("ijlm")
       legs3 = Str("ijlxm")
       legs4 = Str("iyzjlxm")

       signs2 = "-++-"
       signs3 = "-++0-"
       signs4 = "-00++0-"

       idx2 = [1,2,3] 
       idx3 = [1,2,3,4] 
       idx4 = [1,2,4,5,6] 

       # Test truncate
       out = tn.truncate(legs1)  
       assert out == [Str("IJL"), Str("NOJ"), Str("INO")]

       out = tn.truncate(legs2)  
       assert out == Str("ijl")


       # Test cut_unsigned for legs
       out = tn.cut_unsigned(legs2, signs2)
       assert out == Str("ijlm")

       out = tn.cut_unsigned(legs3, signs3)
       assert out == Str("ijlm")

       out = tn.cut_unsigned(legs4, signs4)
       assert out == Str("ijlm")


       # Test cut_unsigned for indices
       out = tn.cut_unsigned(idx2, signs2)
       self.assertList(out, idx2)

       out = tn.cut_unsigned(idx3, signs3)
       self.assertList(out, [1,2,3])

       out = tn.cut_unsigned(idx4, signs4)
       self.assertList(out, [2,3])


       # Test make_symlegs
       out = tn.make_symlegs(legs2, signs2)
       assert out == Str("IJLM")

       out = tn.make_symlegs(legs3, signs3)
       assert out == Str("IJLM")

       out = tn.make_symlegs(legs4, signs4)
       assert out == Str("IJLM")


       # Test get_num_legs
       out = tn.get_num_legs(legs1)
       assert out == 6


       # Test make_legs
       out = tn.make_legs(5)
       assert out == Str("abcde")

       out = tn.make_legs(21)
       assert out == Str("abcdefghijklmnoprstuv")
       assert Str("q") not in out
       assert Str("a")     in out



   def test_generators(self):

       def _test(legs, items, ans_compress, ans_shared, ans_shared_idx):

           a = (legs["A"], items["A"])
           b = (legs["B"], items["B"])

           # Test zip_compress
           out = tn.zip_compress(a, b, legs["C"])
           out = list(out)
           self.assertList(out, ans_compress)

           # Test zip_shared
           out = tn.zip_shared(a, b)
           out = list(out)
           self.assertList(out, ans_shared)

           # Test get_shared_indices
           out = tn.get_shared_indices(a[0], b[0])
           self.assertList(out, ans_shared_idx)


       # Test-1
       legs1  = {"A": Str("IJLM"), "B": Str("NOJL"), "C": Str("INOM")  }
       items1 = {"A": [1,2,3,4],   "B": [5,6,21,31], "C": [1,5,6,4]    }
      
       ans_compress   = [1,5,6,4]
       ans_shared     = [(2,21), (3,31)]
       ans_shared_idx = ([1,2],  [2,3])

       _test(legs1, items1, ans_compress, ans_shared, ans_shared_idx)
       

       # Test-2
       legs2  = {"A": Str("IxzJLM"), \   
                 "B": Str("NOyJxL"), \
                 "C": Str("INOMyz")  }

       items2 = {"A": [1, '*',  '#',   2,     3,    4], \
                 "B": [5,   6,  '!',  21,  '**',   31], \ 
                 "C": [1,   5,    6,   4,   '!',  '#']  }

       ans_compress   = [1,5,6,4,'!','#']
       ans_shared     = [(2,21), (3,31), ('*','**')]
       ans_shared_idx = ([3,4,1], [3,5,4])

       _test(legs2, items2, ans_compress, ans_shared, ans_shared_idx)
      
 

   def test_iterables(self):

       # Test dictzip
       out = tn.dictzip(("A", "B", "C"), ["1234", "5623", "1564"])
       ans = {"A": "1234", "B": "5623", "C": "1564"}
       self.assertDict(out, ans)

       # Test dictriplet
       out = tn.dictriplet("1234", "5623", "1564")
       ans = {"A": "1234", "B": "5623", "C": "1564"}
       self.assertDict(out, ans)

       # Test get_items
       out = tn.get_items(["A","B","C","D","E","F","G"], (1,2,4,6))
       ans = ["B","C","E","G"]
       self.assertList(out, ans)

       # Test to_string
       out = tn.to_string(range(5))
       assert out == "1234"

       # Test sorted_string
       assert tn.sorted_string("MINKJ")     == "IJKMN"
       assert tn.sorted_string("cadbMINKJ") == "IJKMNabcd"

       # Test sort
       lst = [Str("molw"), Str("IJK"), Str("IMNO"), \
              Str("POMpoml"), Str("PQ"), Str("PQL"), Str("ABC")]

       ans = [Str("PQ"), Str("ABC"), Str("IJK"), \
              Str("PQL"), Str("IMNO"), Str("molw"), Str("POMpoml")]   

       out = tn.sort(lst)
       self.assertList(out, ans)



   def test_iterables_with_legs(self):
       
       # Dummy object with legs
       class Box:

          def __init__(self, a, legs):
              self.a    = a
              self.legs = legs

          def __eq__(self, other):
              return (self.a == other.a) and (self.legs == other.legs)


       # Make a list of legs and boxes
       legs_lst = [Str("molw"), Str("IJK"), Str("IMNO"), \
                   Str("POMpoml"), Str("PQ"), Str("PQL"), Str("ABC")]

       boxes = [Box(i, legs) for i, legs in enumerate(legs_lst)]


       # Test get_legs
       out = tn.get_legs(boxes)
       self.assertList(out, legs_lst)

       # Test get_from_legs
       out = tn.get_from_legs(boxes, Str("IMNO"))
       assert out == boxes[2]

       out = tn.get_from_legs(boxes, Str("molw"))
       assert out == boxes[0]

       out = tn.get_from_legs(boxes, Str("POMpoml"))
       assert out == boxes[3]

       # Test sort_by_legs
       out = tn.sort_by_legs(boxes)
       ans = [boxes[i] for i in (4,6,1,5,2,0,3)]
       self.assertList(out, ans)
        

























































































































































































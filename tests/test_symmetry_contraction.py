#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import helper_lib as lib
import taishoten as tn

from taishoten import Str
from .util     import TaishoTenTestCase, must_fail
from .         import util


class TestSymmetryContraction(TaishoTenTestCase):

   # --- Test constructor --------------------------------------------------- #

   def test_construct(self):

       def _test(symA, symB, legs, symlegs, phase):

           sym = tn.dictriplet(symA, symB, None)
           out = tn.SymmetryContraction(symA, symB, legs)
           self.assertSymmetryContraction(out, sym, legs, symlegs, phase)   

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       # Test-1
       symA    = tn.Symmetry1D("++-", (Si,Sj,Sk))
       symB    = tn.Symmetry1D("++-", (Sk,Sl,Sm))
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), None)
       phase   = 1

       _test(symA, symB, legs, symlegs, phase)

       # Test-2
       symA    = tn.Symmetry1D("+++-", (Si,Sn,So,Sm))
       symB    = tn.Symmetry1D("++-",  (Si,Sn,Sp))
       legs    = tn.dictriplet(Str("inom"), Str("inp"), Str("pom"))
       symlegs = tn.dictriplet(Str("INOM"), Str("INP"), None)
       phase = -1

       _test(symA, symB, legs, symlegs, phase)

       # Test-2.1
       symA    = tn.Symmetry1D("+++-", (Si,Sn,So,Sm))
       symB    = tn.Symmetry1D("+--",  (Si,Sn,Sp))
       legs    = tn.dictriplet(Str("inom"), Str("inp"), Str("pom"))
       symlegs = tn.dictriplet(Str("INOM"), Str("INP"), None)
       phase = -1

       must_fail(_test)(symA, symB, legs, symlegs, phase)

       # Test-3
       symA    = tn.Symmetry1D("+0+0+-", (Si,Sn,So,Sm))
       symB    = tn.Symmetry1D("++00-",  (Si,Sn,Sp))
       legs    = tn.dictriplet(Str("ixnyom"), Str("inzxp"), Str("pyzom"))
       symlegs = tn.dictriplet(Str("INOM"),   Str("INP"),   None)
       phase = -1

       _test(symA, symB, legs, symlegs, phase)        

       # Test-4      
       symA    = tn.Symmetry1D("++-", (Si,Sj,Sk), qtot=1, mod=3)
       symB    = tn.Symmetry1D("++-", (Sk,Sl,Sm), qtot=2, mod=3)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), None)
       phase   = 1

       _test(symA, symB, legs, symlegs, phase)   

       # Test-5      
       symA    = tn.Symmetry1D("++-", (Si,Sj,Sk), mod=3)
       symB    = tn.Symmetry1D("++-", (Sk,Sl,Sm))
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), None)
       phase   = 1

       _test(symA, symB, legs, symlegs, phase)     

       # Test-6      
       symA    = tn.Symmetry1D("++-", (Si,Sj,Sk))
       symB    = tn.Symmetry1D("++-", (Sk,Sl,Sm), mod=3)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), None)
       phase   = 1

       _test(symA, symB, legs, symlegs, phase)    



   # --- Test compute() ----------------------------------------------------- #

   def test_compute(self):

       def _test(symA, symB, symC, legs, symlegs, phase):

           sym = tn.dictriplet(symA, symB, symC)
           out = tn.SymmetryContraction(symA, symB, legs)
           out.compute()
           self.assertSymmetryContraction(out, sym, legs, symlegs, phase)        

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       # Test-1
       symA    = tn.Symmetry1D("++-",  (Si,Sj,Sk))
       symB    = tn.Symmetry1D("++-",  (Sk,Sl,Sm))
       symC    = tn.Symmetry1D("+++-", (Si,Sj,Sl,Sm))
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), Str("IJLM"))
       phase   = 1

       _test(symA, symB, symC, legs, symlegs, phase)

       # Test-2
       symA    = tn.Symmetry1D("+++-", (Si,Sn,So,Sm))
       symB    = tn.Symmetry1D("++-",  (Si,Sn,Sp))
       symC    = tn.Symmetry1D("--+",  (Sp,So,Sm))
       legs    = tn.dictriplet(Str("inom"), Str("inp"), Str("pom"))
       symlegs = tn.dictriplet(Str("INOM"), Str("INP"), Str("POM"))
       phase = -1

       _test(symA, symB, symC, legs, symlegs, phase)

       # Test-3
       symA    = tn.Symmetry1D("+0+0+-", (Si,Sn,So,Sm))
       symB    = tn.Symmetry1D("++00-",  (Si,Sn,Sp))
       symC    = tn.Symmetry1D("-00-+",  (Sp,So,Sm))
       legs    = tn.dictriplet(Str("ixnyom"), Str("inzxp"), Str("pyzom"))
       symlegs = tn.dictriplet(Str("INOM"),   Str("INP"),   Str("POM"))
       phase = -1

       _test(symA, symB, symC, legs, symlegs, phase)        

       # Test-4      
       symA    = tn.Symmetry1D("++-",  (Si,Sj,Sk),    qtot=1, mod=3)
       symB    = tn.Symmetry1D("++-",  (Sk,Sl,Sm),    qtot=2, mod=3)
       symC    = tn.Symmetry1D("+++-", (Si,Sj,Sl,Sm), qtot=3, mod=3)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), Str("IJLM"))
       phase   = 1

       _test(symA, symB, symC, legs, symlegs, phase)   

       # Test-4.1      
       symA    = tn.Symmetry1D("++-",  (Si,Sj,Sk),    qtot=1, mod=3)
       symB    = tn.Symmetry1D("-+-",  (Sk,Sl,Sm),    qtot=2, mod=3)
       symC    = tn.Symmetry1D("--+-", (Si,Sj,Sl,Sm), qtot=1, mod=3)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), Str("IJLM"))
       phase   = -1

       _test(symA, symB, symC, legs, symlegs, phase)

       # Test-5      
       symA    = tn.Symmetry1D("++-",  (Si,Sj,Sk),    mod=2)
       symB    = tn.Symmetry1D("++-",  (Sk,Sl,Sm))
       symC    = tn.Symmetry1D("+++-", (Si,Sj,Sl,Sm), mod=2)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), Str("IJLM"))
       phase   = 1

       _test(symA, symB, symC, legs, symlegs, phase)     

       # Test-6      
       symA    = tn.Symmetry1D("++-",  (Si,Sj,Sk))
       symB    = tn.Symmetry1D("++-",  (Sk,Sl,Sm),    mod=5)
       symC    = tn.Symmetry1D("+++-", (Si,Sj,Sl,Sm), mod=5)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
       symlegs = tn.dictriplet(Str("IJK"), Str("KLM"), Str("IJLM"))
       phase   = 1

       _test(symA, symB, symC, legs, symlegs, phase)   



   # --- Test output qtot, mod, symlabels, fullsigns ------------------------ #

   def test_output_qtot(self):

       def _test(symA, symB, legs, ans):

           symcon = tn.SymmetryContraction(symA, symB, legs)
           out    = symcon.output_qtot()
           assert out == ans

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       # Test-1
       symA = tn.Symmetry1D("++-", (Si,Sj,Sk))
       symB = tn.Symmetry1D("++-", (Sk,Sl,Sm))
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       _test(symA, symB, legs, 0)

       # Test-4      
       symA    = tn.Symmetry1D("++-", (Si,Sj,Sk), qtot=1, mod=3)
       symB    = tn.Symmetry1D("++-", (Sk,Sl,Sm), qtot=2, mod=3)
       legs    = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       _test(symA, symB, legs, 3)

       # Test-4.1      
       symA = tn.Symmetry1D("++-", (Si,Sj,Sk), qtot=1, mod=3)
       symB = tn.Symmetry1D("-+-", (Sk,Sl,Sm), qtot=2, mod=3)
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))
        
       _test(symA, symB, legs, 1)



   def test_output_mod(self):

       def _test(symA, symB, legs, ans):

           symcon = tn.SymmetryContraction(symA, symB, legs)
           out    = symcon.output_mod()

           if   ans is None:
                assert out is None
           else:
                assert out == ans

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       # Test-1
       symA = tn.Symmetry1D("++-", (Si,Sj,Sk))
       symB = tn.Symmetry1D("++-", (Sk,Sl,Sm))
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       _test(symA, symB, legs, None)

       # Test-4      
       symA = tn.Symmetry1D("++-", (Si,Sj,Sk), qtot=1, mod=3)
       symB = tn.Symmetry1D("++-", (Sk,Sl,Sm), qtot=2, mod=3)
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       _test(symA, symB, legs, 3)   

       # Test-5      
       symA = tn.Symmetry1D("++-", (Si,Sj,Sk), mod=2)
       symB = tn.Symmetry1D("++-", (Sk,Sl,Sm))
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       _test(symA, symB, legs, 2)    

       # Test-6      
       symA = tn.Symmetry1D("++-", (Si,Sj,Sk))
       symB = tn.Symmetry1D("++-", (Sk,Sl,Sm), mod=5)
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       _test(symA, symB, legs, 5)

       # Extra test
       symA = tn.Symmetry1D("++-", (Si,Sj,Sk), mod=3)
       symB = tn.Symmetry1D("++-", (Sk,Sl,Sm), mod=5)
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       must_fail(_test)(symA, symB, legs, None)



   def test_output_symlabels(self):

       def _test(symA, symB, legs, fullsigns, ans):

           symcon = tn.SymmetryContraction(symA, symB, legs)
           symcon.compute()

           out = symcon.output_symlabels(fullsigns)           
           self.assertEqualArrayList(out, ans)       

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       # Test-1
       symA = tn.Symmetry1D("++-",  (Si,Sj,Sk))
       symB = tn.Symmetry1D("++-",  (Sk,Sl,Sm))
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       fullsigns = "+++-"
       symlabels = (Si,Sj,Sl,Sm)

       _test(symA, symB, legs, fullsigns, symlabels)

       # Test-2
       symA = tn.Symmetry1D("+++-", (Si,Sn,So,Sm))
       symB = tn.Symmetry1D("++-",  (Si,Sn,Sp))
       legs = tn.dictriplet(Str("inom"), Str("inp"), Str("pom"))

       fullsigns = "--+"
       symlabels = (Sp,So,Sm)

       _test(symA, symB, legs, fullsigns, symlabels)

       # Test-3
       symA = tn.Symmetry1D("+0+0+-", (Si,Sn,So,Sm))
       symB = tn.Symmetry1D("++00-",  (Si,Sn,Sp))
       legs = tn.dictriplet(Str("ixnyom"), Str("inzxp"), Str("pyzom"))

       fullsigns = "-00-+"
       symlabels = (Sp,So,Sm)

       _test(symA, symB, legs, fullsigns, symlabels)

       # Test-4      
       symA = tn.Symmetry1D("++-", (Si,Sj,Sk), qtot=1, mod=3)
       symB = tn.Symmetry1D("++-", (Sk,Sl,Sm), qtot=2, mod=3)
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       fullsigns = "+++-"
       symlabels = (Si,Sj,Sl,Sm)

       _test(symA, symB, legs, fullsigns, symlabels)


       
   def test_output_fullsigns(self):

       def _test(symA, symB, legs, ans):

           symcon = tn.SymmetryContraction(symA, symB, legs)
           symcon.compute()

           out = symcon.output_fullsigns()           
           self.assertEqualArrayList(out, ans)       

       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       # Test-1
       symA = tn.Symmetry1D("++-",  (Si,Sj,Sk))
       symB = tn.Symmetry1D("++-",  (Sk,Sl,Sm))
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       _test(symA, symB, legs, "+++-")

       # Test-2
       symA = tn.Symmetry1D("+++-", (Si,Sn,So,Sm))
       symB = tn.Symmetry1D("++-",  (Si,Sn,Sp))
       legs = tn.dictriplet(Str("inom"), Str("inp"), Str("pom"))

       _test(symA, symB, legs, "--+")

       # Test-3
       symA = tn.Symmetry1D("+0+0+-", (Si,Sn,So,Sm))
       symB = tn.Symmetry1D("++00-",  (Si,Sn,Sp))
       legs = tn.dictriplet(Str("ixnyom"), Str("inzxp"), Str("pyzom"))

       _test(symA, symB, legs, "-00-+")

       # Test-4      
       symA = tn.Symmetry1D("++-", (Si,Sj,Sk), qtot=1, mod=3)
       symB = tn.Symmetry1D("++-", (Sk,Sl,Sm), qtot=2, mod=3)
       legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

       _test(symA, symB, legs, "+++-")



   # --- Test align_symlabels() --------------------------------------------- #

   def test_align_symlabels(self):

       Si = np.arange(0,5) 
       Sj = np.arange(0,2)
       Sk = np.arange(1,5)
       Sl = np.arange(0,4)
       Sm = np.arange(1,5)
       Sn = np.arange(0,5)
       So = np.arange(0,2)
       Sp = np.arange(0,5)

       def _test(symlabels_A, symlabels_B, modA, modB, ans, ans1):

           symA = tn.Symmetry1D("++-", (Si,Sj,Sk), mod=modA)
           symB = tn.Symmetry1D("++-", (Sk,Sl,Sm), mod=modB)
           legs = tn.dictriplet(Str("ijk"), Str("klm"), Str("ijlm"))

           symcon = tn.SymmetryContraction(symA, symB, legs)
           out    = symcon.align_symlabels(symlabels_A, symlabels_B)

           self.assertEqualArray(out, ans)
           self.assertEqualArray(out, ans1)

       # Test-1
       ans  = lib.align_symlabels(Si, Sj)
       ans1 = np.array([0,1])
       _test(Si, Sj, None, None, ans, ans1)

       # Test-2
       ans  = lib.align_symlabels(Si, Sk)
       ans1 = np.array([1,2,3,4])
       _test(Si, Sk, 3, 3, ans, ans1)

       # Test-3
       ans  = lib.align_symlabels(Sk, lib.fold_1D(Si, 3)) 
       ans1 = np.array([1,2])
       _test(Si, Sk, None, 3, ans, ans1)

       # Test-4
       ans  = lib.align_symlabels(Si, lib.fold_1D(Sk, 3)) 
       ans1 = np.array([0,1,2])
       _test(Si, Sk, 3, None, ans, ans1)



   # --- Test auxiliary symmetry -------------------------------------------- #

   def test_compute_aux_symmetry(self):

       def _test(symA, symB, legs, signs, symlabels):

           symcon = tn.SymmetryContraction(symA, symB, legs)
           out    = tn.symmetry.compute_aux_symmetry(symcon)

           self.assertSymmetry(out, signs, symlabels)  

       # Define symlabels
       Si = np.arange(0,5) 
       Sj = np.arange(0,2)
       Sk = np.arange(1,5)
       Sl = np.arange(0,4)
       Sm = np.arange(1,5)
       Sn = np.arange(0,5)
       So = np.arange(0,2)
       Sp = np.arange(0,5)

       # Test-1
       symA = tn.Symmetry1D("+++-", (Si,Sn,So,Sm))
       symB = tn.Symmetry1D("++-",  (Si,Sn,Sp))
       legs = tn.dictriplet(Str("inom"), Str("inp"), Str("pom"))

       Saux_A = lib.make_aux_symlabels(("++", "+-"), ([Si,Sn], [So,Sm]))
       Saux_B = lib.make_aux_symlabels(("--", "+"),  ([Si,Sn], [Sp]))
       Saux   = lib.align_symlabels(Saux_A, Saux_B)

       _test(symA, symB, legs, "++-", [Si,Sn,Saux])

       # Test-2
       symA = tn.Symmetry1D("+0+0+-", (Si,Sn,So,Sm))
       symB = tn.Symmetry1D("++00-",  (Si,Sn,Sp))
       legs = tn.dictriplet(Str("ixnyom"), Str("inzxp"), Str("pyzom"))

       _test(symA, symB, legs, "++-", [Si,Sn,Saux])

       

   # --- Test map contractions ---------------------------------------------- #

   def test_compute_maps(self):

       # Define symmetries
       Si = np.arange(0,5) 
       Sj = np.arange(0,2)
       Sk = np.arange(1,5)
       Sl = np.arange(0,4)
       Sm = np.arange(1,5)
       Sn = np.arange(0,5)
       So = np.arange(0,2)
       Sp = np.arange(0,5)

       Saux_A  = lib.make_aux_symlabels(("++", "+-"), ([Si,Sn], [So,Sm]))
       Saux_B  = lib.make_aux_symlabels(("--", "+"),  ([Si,Sn], [Sp]))
       Saux    = lib.align_symlabels(Saux_A, Saux_B)

       symA    = tn.Symmetry1D("+++-", (Si,Sn,So,Sm))
       symB    = tn.Symmetry1D("++-",  (Si,Sn,Sp))
       aux_sym = tn.Symmetry1D("++-",  (Si,Sn,Saux))

       # Compute map list
       mapA = tn.Map.compute(symA,    Str("INOM")) 
       mapB = tn.Map.compute(symB,    Str("INP"))
       mapQ = tn.Map.compute(aux_sym, Str("INQ"))
       maps = [mapA, mapB, mapQ]

       AB = np.einsum("INOM,INP->MOP", mapA.array, mapB.array)
       AQ = np.einsum("INOM,INQ->MOQ", mapA.array, mapQ.array)
       BQ = np.einsum("INP,INQ->PQ",   mapB.array, mapQ.array)

       mapAB = tn.Map(AB, Str("MOP"))
       mapAQ = tn.Map(AQ, Str("MOQ"))
       mapBQ = tn.Map(BQ, Str("PQ"))

       ans = [mapA, mapB, mapQ, mapAB, mapAQ, mapBQ]

       # Get output from compute_maps()
       out = cp.deepcopy(maps)
       out = tn.symmetry.compute_maps(out)
       out = sorted(out)

       # Test
       self.assertList(out, ans, fun=self.assertEqualMap)

       self.assertMap(out[0], Str("INOM"), (5,5,2,4))
       self.assertMap(out[1], Str("INP"),  (5,5,5))
       self.assertMap(out[2], Str("INQ"),  (5,5,len(Saux)))
       self.assertMap(out[3], Str("MOP"),  (4,2,5))
       self.assertMap(out[4], Str("MOQ"),  (4,2,len(Saux)))
       self.assertMap(out[5], Str("PQ"),   (5,len(Saux)))



   def test_compute_pairwise_contractions_of_maps(self):

       # Define symmetries
       Si = range(0,5) 
       Sj = range(0,2)
       Sk = range(1,5)
       Sl = range(0,4)
       Sm = range(1,5)
       Sn = range(0,5)
       So = range(0,2)
       Sp = range(0,5)

       symA = tn.Symmetry1D("+++-", (Si,Sn,So,Sm))
       symB = tn.Symmetry1D("++-",  (Si,Sn,Sp))

       # Compute map list
       mapA = lib.create_random_map(Str("INOM"), (5,5,2,4), 14) 
       mapB = lib.create_random_map(Str("INP"),  (5,5,5),   10)
       mapQ = lib.create_random_map(Str("INQ"),  (5,5,5),   10)
       maps = [mapA, mapB, mapQ]

       AB = np.einsum("INOM,INP->MOP", mapA.array, mapB.array)
       AQ = np.einsum("INOM,INQ->MOQ", mapA.array, mapQ.array)
       BQ = np.einsum("INP,INQ->PQ",   mapB.array, mapQ.array)

       mapAB = tn.Map(AB, Str("MOP"))
       mapAQ = tn.Map(AQ, Str("MOQ"))
       mapBQ = tn.Map(BQ, Str("PQ"))

       ans = [mapA, mapB, mapQ, mapAB, mapAQ, mapBQ]

       # Get output from compute_pairwise_contractions_of_maps()
       out = cp.deepcopy(maps)
       out = tn.symmetry.compute_pairwise_contractions_of_maps(out)
       out = sorted(out)

       # Test
       self.assertList(out, ans, fun=self.assertEqualMap)

       self.assertMap(out[0], Str("INOM"), (5,5,2,4))
       self.assertMap(out[1], Str("INP"),  (5,5,5))
       self.assertMap(out[2], Str("INQ"),  (5,5,5))
       self.assertMap(out[3], Str("MOP"),  (4,2,5))
       self.assertMap(out[4], Str("MOQ"),  (4,2,5))
       self.assertMap(out[5], Str("PQ"),   (5,5))
       
       
       























































































































#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import helper_lib as lib
import taishoten as tn

from taishoten import Str
from .util     import TaishoTenTestCase



def setup_tensor(*args, **kwargs):
    return lib.setup_tensor_3D(*args, **kwargs)[0]



class TestSymeinsum(TaishoTenTestCase):

   def setUp(self):

       self.mod  = (2 * np.pi / 5) * np.eye(3)
       self.kpts = lib.make_symlabels_3D(mod, [2,2,1])


   def test_222(self):

       np.random.seed(1)

       A = setup_tensor("+-", [self.kpts]*2, (3,3), mod=self.mod)
       B = setup_tensor("+-", [self.kpts]*2, (3,5), mod=self.mod)

       sub  = "ki,ka->ia"
       sub1 = "Kiki,KAka->IAia"
       sub2 = "IAia,IA->Iia"
       self.assertSymeinsum(A, B, sub, sub1, sub2)


   def test_224(self):

       np.random.seed(1)

       A = setup_tensor("+-", [self.kpts]*2, (3,3), mod=self.mod)
       B = setup_tensor("+-", [self.kpts]*2, (3,5), mod=self.mod)

       sub  = "ia,jb->ijab"
       sub1 = "IAia,JBjb->IJABijab"
       sub2 = "IJABijab,IJAB->IJAijab"
       self.assertSymeinsum(A, B, sub, sub1, sub2)


   def test_424(self):

       np.random.seed(1)

       A = setup_tensor("+-+-", [self.kpts]*4, (3,5,3,3), mod=self.mod)
       B = setup_tensor("+-",   [self.kpts]*2, (3,5),     mod=self.mod)

       sub  = "kclj,ic->klij"
       sub1 = "KCLJkclj,ICic->KLIJklij"
       sub2 = "KLIJklij,KLIJ->KLIklij"
       self.assertSymeinsum(A, B, sub, sub1, sub2)


   def test_442(self):

       np.random.seed(1)

       A = setup_tensor("+-+-", [self.kpts]*4, (3,5,3,5), mod=self.mod)
       B = setup_tensor("++--", [self.kpts]*4, (3,3,5,5), mod=self.mod)

       sub  = "kcld,ilcd->ki"
       sub1 = "KCLDkcld,ILCDilcd->KIki"
       sub2 = "KIki,KI->Kki"
       self.assertSymeinsum(A, B, sub, sub1, sub2)


   def test_444(self):

       np.random.seed(1)

       A = setup_tensor("+-+-", [self.kpts]*4, (3,5,3,5), mod=self.mod)
       B = setup_tensor("++--", [self.kpts]*4, (3,3,5,5), mod=self.mod)

       sub  = "kcld,ijcd->klij"
       sub1 = "KCLDkcld,IJCDijcd->KLIJklij"
       sub2 = "KLIJklij,KLIJ->KLIklij"
       self.assertSymeinsum(A, B, sub, sub1, sub2)


   def test_343(self):

       np.random.seed(1)

       A = setup_tensor("++--", [self.kpts]*4, (3,3,3,3), mod=self.mod)
       B = setup_tensor("++-",  [self.kpts]*3, (3,3,5),   mod=self.mod, \
                                                          qtot=self.kpts[2])

       sub  = "klij,klb->ijb"
       sub1 = "KLIJklij,KLBklb->IJBijb"
       sub2 = "IJBijb,IJB->IJijb"
       self.assertSymeinsum(A, B, sub, sub1, sub2)


   def test_431(self):

       np.random.seed(1)

       A = setup_tensor("++--", [self.kpts]*4, (3,3,5,5), mod=self.mod)
       B = setup_tensor("++-",  [self.kpts]*3, (3,3,5),   mod=self.mod, \
                                                          qtot=self.kpts[2])
       sub  = "lkdc,kld->c"
       sub1 = "LKDClkdc,KLDkld->Cc"
       sub2 = "Cc,C->c"
       self.assertSymeinsum(A, B, sub, sub1, sub2)


   def test_440(self):

       np.random.seed(1)

       A = setup_tensor("++--", [self.kpts]*4, (3,3,5,5), mod=self.mod)
       B = setup_tensor("++--", [self.kpts]*4, (3,3,5,5), mod=self.mod)

       out = tn.symeinsum("ijab,ijba->", A, B)

       arrayA = A.get_full_array()
       arrayB = B.get_full_array()
       arrayC = np.einsum("IJABijab,IJABijba->", arrayA, arrayB)

       self.assertTensorVsArray(out, arrayC)


   def test_233(self):

       np.random.seed(1)

       A = setup_tensor("+-",  [self.kpts]*2, (5,5),   mod=self.mod)
       B = setup_tensor("++-", [self.kpts]*3, (3,3,5), mod=self.mod, \
                                                       qtot=self.kpts[2])
       sub  = "bd,ijd->ijb"
       sub1 = "BDbd,IJDijd->IJBijb"
       sub2 = "IJBijb,IJB->IJijb"
       self.assertSymeinsum(A, B, sub, sub1, sub2)


   def test_143(self):

       np.random.seed(1)

       A = setup_tensor("+",    [self.kpts],   (5,),      mod=self.mod, \
                                                          qtot=self.kpts[2])
       B = setup_tensor("++--", [self.kpts]*4, (3,3,5,5), mod=self.mod)

       sub  = "c,ijcb->ijb"
       sub1 = "Cc,IJCBijcb->IJBijb"
       sub2 = "IJBijb,IJB->IJijb"
       self.assertSymeinsum(A, B, sub, sub1, sub2)

          









































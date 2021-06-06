#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import helper_lib as lib
import taishoten as tn

from taishoten import Str
from .util     import TaishoTenTestCase



def setup_tensor(*args, **kwargs):
    return lib.setup_tensor(*args, **kwargs)[0]



class TestSymeinsum(TaishoTenTestCase):


   def test_334(self):

       np.random.seed(1)

       Si = np.arange(0,5) 
       Sj = np.arange(0,2)
       Sk = np.arange(1,5)
       Sl = np.arange(0,4) 
       Sm = np.arange(1,5)

       A = setup_tensor("++-", (Si,Sj,Sk), (8,8,8))
       B = setup_tensor("++-", (Sk,Sl,Sm), (8,8,8))

       sub  = "ijk,klm->ijlm"
       sub1 = "IJKijk,KLMklm->IJLMijlm"
       sub2 = "IJLMijlm,IJLM->IJLijlm"
       self.assertSymeinsum(A, B, sub, sub1, sub2)

       

   def test_444(self):

       np.random.seed(1)

       Si = np.arange(0,5) 
       Sj = np.arange(0,2)
       Sl = np.arange(0,4) 
       Sm = np.arange(1,5)
       Sn = np.arange(0,5)
       So = np.arange(0,2)

       A = setup_tensor("+++-", (Si,Sj,Sl,Sm), (8,8,8,8))
       B = setup_tensor("++--", (Sn,So,Sj,Sl), (8,8,8,8))

       sub  = "ijlm,nojl->inom"
       sub1 = "IJLMijlm,NOJLnojl->INOMinom"
       sub2 = "INOMinom,INOM->INOinom"
       self.assertSymeinsum(A, B, sub, sub1, sub2)



   def test_433(self):

       np.random.seed(1)

       Si = np.arange(0,5) 
       Sj = np.arange(0,2)
       Sl = np.arange(0,4) 
       Sm = np.arange(1,5)
       Sn = np.arange(0,5)
       So = np.arange(0,2)
       Sp = np.arange(0,5)

       A = setup_tensor("+++-", (Si,Sn,So,Sm), (8,8,8,8))
       B = setup_tensor("++-",  (Si,Sn,Sp),    (8,8,8))

       sub  = "inom,inp->pom"
       sub1 = "INOMinom,INPinp->POMpom"
       sub2 = "POMpom,POM->POpom"
       self.assertSymeinsum(A, B, sub, sub1, sub2)



   def test_453(self):

       np.random.seed(1)

       Si = np.arange(0,6) 
       Sj = np.arange(0,3)
       Sk = np.arange(1,6) 
       Sl = np.arange(0,3) 
       Sm = np.arange(0,3)
       Sn = np.arange(0,6)

       A = setup_tensor("++--",  (Si,Sj,Sk,Sl),    (8,8,8,8))
       B = setup_tensor("-++--", (Sj,Sk,Sl,Sm,Sn), (8,8,8,8,8))

       sub  = "ijkl,jklmn->imn"
       sub1 = "IJKLijkl,JKLMNjklmn->IMNimn"
       sub2 = "IMNimn,IMN->IMimn"
       self.assertSymeinsum(A, B, sub, sub1, sub2)



   def test_unsigned_kron_555(self):

       np.random.seed(1)

       Si = np.arange(0,2) 

       A = setup_tensor("++-0-", (Si,Si,Si,Si), (5,5,8,3,8))
       B = setup_tensor("++-0-", (Si,Si,Si,Si), (8,8,8,3,8))

       sub  = "ijcxd,abcyd->ijxyab"
       sub1 = "IJCDijcxd,ABCDabcyd->IJABijxyab"
       sub2 = "IJABijxyab,IJAB->IJAijxyab"
       self.assertSymeinsum(A, B, sub, sub1, sub2)



   def test_unsigned_dot_554(self):

       np.random.seed(1)

       Si = np.arange(0,2) 

       A = setup_tensor("++-0-", (Si,Si,Si,Si), (5,5,8,3,8))
       B = setup_tensor("++-0-", (Si,Si,Si,Si), (8,8,8,3,8))

       sub  = "ijcxd,abcxd->ijab"
       sub1 = "IJCDijcxd,ABCDabcxd->IJABijab"
       sub2 = "IJABijab,IJAB->IJAijab"
       self.assertSymeinsum(A, B, sub, sub1, sub2)



   def test_unsigned_hadamard_555(self):

       np.random.seed(1)

       Si = np.arange(0,2) 

       A = setup_tensor("++--0", (Si,Si,Si,Si), (5,5,8,3,8))
       B = setup_tensor("++-0-", (Si,Si,Si,Si), (8,8,8,3,8))

       sub  = "ijcdx,abcxd->ijabx"
       sub1 = "IJCDijcdx,ABCDabcxd->IJABijabx"
       sub2 = "IJABijabx,IJAB->IJAijabx"
       self.assertSymeinsum(A, B, sub, sub1, sub2)





























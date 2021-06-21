#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import copy  as cp
import numpy as np

import lib
import util

import taishoten as tn
from taishoten import Str
from taishoten.transformations import Node








@pytest.fixture
def fixt_symeinsum_3D(fixt_tensors_3D):

    dct = {}
    def make(keyA, keyB, sub, sub1, sub2, idx=1):

        A, _ = fixt_tensors_3D[keyA]
        B, _ = fixt_tensors_3D[keyB]

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
   def request_symeinsum(self, fixt_symeinsum_3D):
       self._symeinsum_data = fixt_symeinsum_3D


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

























































































































































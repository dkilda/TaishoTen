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
def fixt_symeinsum_1D(fixt_tensors_1D):

    dct = {}
    def make(keyA, keyB, sub, sub1, sub2, idx=1):

        A, _ = fixt_tensors_1D[keyA]
        B, _ = fixt_tensors_1D[keyB]

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
   def request_symeinsum(self, fixt_symeinsum_1D):
       self._symeinsum_data = fixt_symeinsum_1D


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
























































































































































#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import reference_data_helper as helper

from .util import save_array



# --- Auxiliary functions --------------------------------------------------- #

def randn(*args, cmplx=False):
    if   cmplx:
         return randn(*args) + 1.0j*randn(*args)
    else:
         return np.random.randn(*args)


def randint(low=0, high=None, size=None, dtype=int):
    return np.random.randint(low, high=high, size=size, dtype=dtype)


def make_str(x):

    if  isiterable(x):
        return ''.join(str(v) for v in x)
    else:
        return str(x)


def make_filename(name, *args):

    str_args = "_".join(map(make_str, args))
    filename = "{}_{}".format(name, str_args)
    return filename


def make_input_filename(name, *args):
    return make_filename("/input/{}".format(name), *args)


def make_output_filename(name, *args):
    return make_filename("/output/{}".format(name), *args)



# --- Main functions for generating data ------------------------------------ #

def gen_basic_data():

    a = randn(2,3,4)
    save_array("input/a_234_real", a)
    
    b = randn(3,2)
    save_array("input/b_32_real", b)

    c = randn(4,3,2,4, cmplx=True)
    save_array("input/c_4324_complex", c)



def gen_map_data():

    # Data for Map.compute_from_idx
    def gen_compute_from_idx(descr, shape, num_idx, legs):

        name = "map_compute_from_idx"
        fin  = make_input_filename(name,  descr, shape, num_idx)
        fout = make_output_filename(name, descr, shape, num_idx)

        idx = randint(high=np.prod(shape), size=num_idx)
        out = helper.compute_map_from_idx(shape, idx)

        inp = np.array([shape, idx, legs], dtype='object')
        save_array(fin,  inp)
        save_array(fout, out)

        return out

    A = gen_compute_from_idx("A", (2,3,4),   5,  "IJK")
    B = gen_compute_from_idx("B", (3,2),     2,  "IJ")
    C = gen_compute_from_idx("C", (4,3,2,4), 10, "IJKL")


    # Data for Map.compute
    def gen_compute_map(descr, symlabels, signs, qtot=0, mod=None):

        compute_map()
  


    # Data for contract_maps
    def gen_contract_maps(descr, subscript, arrA, arrB):

        fout = make_output_filename("contract_maps", descr, subscript)
        out  = np.einsum(subscript, arrA, arrB)
        save_array(fout, out)
        return out
 

    gen_contract_maps("AB", "IJK,IM->JKM",     A, B)
    gen_contract_maps("AC", "IJKL,MJIN->IKMN", A, C)
    gen_contract_maps("CA", "IJKL,KMN->MNIJL", C, A)




def gen_symmetry_data():
    
    def gen_sym():

        nbond = 8
        ni = range(0,5)
        nj = range(0,2)
        nk = range(1,5)
        nl = range(0,4)
        nm = range(1,5)
        nn = range(0,5)
        no = range(0,2)
        np = range(0,5)

        # FIXME convert each of ni, nj to ndarray

        fullsigns = ['++-']
        symlabels = [ni,nj,nk]

        fullsigns = ['+++-']
        symlabels = [ni,nj,nl,nm]

        fullsigns = ['++--']
        symlabels = [nn,no,nj,nl]


        indices = range(len(symlabels))







        signs = make_signs(fullsigns)



        

        ################################################################

        # FLATTEN

        symshape = (len(s) for s in symlabels)
        symsize  = np.prod(symshape)


        idx       = np.indices(symshape)
        combo_idx = np.ravel_multi_index(idx, symshape)


        flat_symlabels = np.zeros((symsize,))

        nonzero_idx = []

        for i in range(symshape[0]):
          for j in range(symshape[1]):
            for k in range(symshape[2]):

                symlb = symlabels[0][i]*signs[0] 
                      + symlabels[1][j]*signs[1] 
                      + symlabels[2][k]*signs[2]

                flat_symlabels[combo_idx[i,j,k]] = symlb

                if  abs(symlb) < tol:
                    nonzero_idx.append(combo_idx[i,j,k])

        flat_symlabels = flat_symlabels * phase

        out = np.zeros(symshape)
        for idx in nonzero_idx:
            out[*idx] = 1.0



        if   len(symlabels) == 1:
             return symlabels

        elif len(symlabels) == 2:
             return flatted_symlabels_2D(symlabels, signs)

        elif len(symlabels) == 3:
             return flatted_symlabels_3D(symlabels, signs)

        elif len(symlabels) == 4:
             return flatted_symlabels_4D(symlabels, signs)

        elif len(symlabels) == 5:
             return flatted_symlabels_5D(symlabels, signs)


        ################################################################

        # Compute map

        flat_symlabels = flat_symlabels_3D(symlabels, signs)
        flat_symlabels = np.mod(flat_symlabels - qtot, mod)

        zero_idx = []
        for i in range(len(flat_symlabels)):
            if  flat_symlabels[i] < tol:
                zero_idx.append(i)


        map_array = compute_map_from_idx(symshape, zero_idx)




        ################################################################

        # AUX
        
        mid  = flat_symlabels_2D((symlabels[0], symlabels[2]), (signs[0], signs[2]))
        edge = flat_symlabels_2D((symlabels[1], symlabels[3]), (signs[1], signs[3]), -1)

        edge += qtot

        mid  = fold(mid)
        edge = fold(edge)

        aux = align_symlabels(mid, edge)


        ################################################################

        # ALIGN

        zero_idx = []
        combo_symlabels = np.zeros((len(symlabels_11), len(symlabels_22)))

        for i in range(len(symlabels_11)):
          for j in range(len(symlabels_22)):
              combo_symlabels[i,j] = abs(symlabels_11[i] - symlabels_22[j])

              if  combo_symlabels[i,j] < tol:
                  zero_idx.append(i)

        aligned_symlabels_11 = symlabels_11[tuple(zero_idx)]
    
 
        ################################################################

        # FOLD

        if  mod is not None:
            symlabels = np.mod(symlabels, mod)

        symlabels = np.unique(symlabels)


        ################################################################



np.random.seed(1)

gen_basic_data()
gen_map_data()





















































































































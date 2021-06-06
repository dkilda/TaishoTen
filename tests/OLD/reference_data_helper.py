#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def isiterable(x):
    try:
        x_iterator = iter(x)
        isiter = True
    except TypeError:
        isiter = False

    return isiter



def subscript_to_legs(subscript):

    # Remove any spaces, add "->" if not present
    sub = subscript.replace(' ', '')
    if '->' not in sub: sub += '->'
 
    # Split up into list, using ',' as delim
    legs = sub.replace('->', ',').split(',')
    legs = [StrSet(x) for x in legs]
    return legs



def legs_to_subscript(*legs):

    # Convert list of legs to subscript 
    subscript = ','.join(legs[:-1].to_str()) + '->' + legs[-1].to_str() 
    return subscript  



def signs_to_ints(signs):

    # Convert sign string to ints
    signs = [sgn                       for sgn in signs if sgn != '0']
    signs = [(1 if sgn == '+' else -1) for sgn in signs]
    return signs



def get_symlabels_shape(symlabels):

    # Get shape of symlabels
    return (len(s) for s in symlabels)



def flatten_symlabels(symlabels, signs, phase=1):

    # Process signs
    signs = signs_to_ints(signs)
    assert len(symlabels) == len(signs)

    # Get shape and size of symlabels
    shape = get_symlabels_shape(symlabels)
    size  = np.prod(shape)

    # Get idx array (e.g. idx[0][i,j,k] = i, 
    #                     idx[1][i,j,k] = j, 
    #                     idx[2][i,j,k] = k, etc) 
    # and flat_idx array (e.g. flat_idx[i,j,k] = flattened index at i,j,k) TODO compare np.indices and np.meshgrid
    idx      = np.indices(shape)
    flat_idx = np.ravel_multi_index(idx, shape)

    # Compute flattened symlabels for various dims of symlabels
    flat = np.zeros((size,)) 

    if    len(symlabels) == 1:
          pass

    elif  len(symlabels) == 2:

          for i in range(shape[0]):
            for j in range(shape[1]):

                flat[flat_idx[i,j]] = symlabels[0][i] * signs[0] \
                                    + symlabels[1][j] * signs[1] 

    elif  len(symlabels) == 3:

          for i in range(shape[0]):
            for j in range(shape[1]):
              for k in range(shape[2]):

                  flat[flat_idx[i,j,k]] = symlabels[0][i] * signs[0] \
                                        + symlabels[1][j] * signs[1] \
                                        + symlabels[2][k] * signs[2] 

    elif  len(symlabels) == 4:

          for i in range(shape[0]):
            for j in range(shape[1]):
              for k in range(shape[2]):
                for l in range(shape[3]):

                    flat[flat_idx[i,j,k,l]] = symlabels[0][i] * signs[0] \
                                            + symlabels[1][j] * signs[1] \
                                            + symlabels[2][k] * signs[2] \
                                            + symlabels[3][l] * signs[3] 

    elif  len(symlabels) == 5:

          for i in range(shape[0]):
            for j in range(shape[1]):
              for k in range(shape[2]):
                for l in range(shape[3]):
                  for m in range(shape[4]):

                      flat[flat_idx[i,j,k,l,m]] = symlabels[0][i] * signs[0] \
                                                + symlabels[1][j] * signs[1] \
                                                + symlabels[2][k] * signs[2] \
                                                + symlabels[3][l] * signs[3] \
                                                + symlabels[4][m] * signs[4] 

    else:
          raise NotImplementedError("flatten_symlabels: len > 5 not supported")


    # Phase factor
    flat = flat * phase
    return flat



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



def random_map_from_idx(shape, num_idx):

    idx = randint(0, np.prod(shape), num_idx)
    return compute_map_from_idx(shape, idx)




def compute_map_from_idx(shape, idx):

    # Compute map from shape, idx input
    out = np.zeros(shape)
    for i in idx:
        out[i] = 1.0
    return out



def compute_map(signs, symlabels, qtot=0, mod=None, tol=CLOSE_RTOL):

    # Compute map from symmetry info
    flat_symlabels = flatten_symlabels(symlabels, signs)
    flat_symlabels = np.mod(flat_symlabels - qtot, mod)

    if  flat_symlabels.ndim > 1:
        for i in range(len(flat_symlabels)):
            flat_symlabels[i] = sum(abs(flat_symlabels[i]))

    idx   = find_zeros(flat_symlabels, tol=tol)
    shape = get_symlabels_shape(symlabels)    
    out   = compute_map_from_idx(shape, idx)
    return out



def fold_symlabels(symlabels, mod):

    # Apply mod and take unique symlabels
    if  mod is not None:
        symlabels = np.mod(symlabels, mod)

    symlabels = np.unique(symlabels)
    return symlabels



def align_symlabels(symlabels_11, symlabels_22, tol=CLOSE_RTOL):

    # Align symlabels_11 with symlabels_22, by finding idx where
    # the conservation law between them is satisfied
    idx = []
    combo_symlabels = np.zeros((len(symlabels_11), len(symlabels_22)))

    for i in range(len(symlabels_11)):
      for j in range(len(symlabels_22)):

          combo_symlabels[i,j] = abs(symlabels_11[i] - symlabels_22[j])

          if  isiterable(combo_symlabels[i,j]):
              combo_symlabels[i,j] = sum(combo_symlabels[i,j])

          if  combo_symlabels[i,j] < tol:
              idx.append(i)

    aligned_symlabels_11 = symlabels_11[list(idx)]
    return aligned_symlabels_11    



"""
def meshgrid(xs, indices=None):

    # Select vectors corresponding to indices
    # (we'll use these selected vectors to construct meshgrid)
    if IS(indices):
       xs = [xs[i] for i in indices]

    # Construct meshgrid
    grid = np.meshgrid(*xs, indexing='ij')
    return grid
"""
    




def make_aux_symlabels(mid_symlabels,  mid_signs,  \
                       edge_symlabels, edge_signs, \
                       qtot=0, mod=None, phase=1, tol=CLOSE_RTOL):

    # Make auxiliary symlabels
    mid   = flatten_symlabels(mid_symlabels,  mid_signs,     phase)   
    edge  = flatten_symlabels(edge_symlabels, edge_signs, -1*phase)
    edge += qtot

    mid  = fold_symlabels(mid,  mod)
    edge = fold_symlabels(edge, mod)
    aux  = align_symlabels(mid, edge, tol=tol)
    return aux 




def compute_phase(legs_A, signs_A, \
                  legs_B, signs_B):

    # Compute relative phase
    legs_A = ''.join(lg for lg, sgn in zip(legs_A, signs_A) if sgn != '0') 
    legs_B = ''.join(lg for lg, sgn in zip(legs_B, signs_B) if sgn != '0')
    shared = set(legs_A + legs_B)

    signs_A = ''.join(sgn for sgn in signs_A if sgn != '0')
    signs_B = ''.join(sgn for sgn in signs_B if sgn != '0')

    phases = []
    for lg in shared:

        iA = legs_A.find(lg)
        iB = legs_B.find(lg)

        if   signs_A[iA] == signs_B[iB]:
             ph = -1
        else:
             ph = 1

        phases.append(ph)

    assert np.unique(phases).size == 1  
    phase = phases[0]
    return phase




def compute_output_symlabels(subscript, symlabels_A, symlabels_B):

    # Compute output symlabels
    legs   = subscript_to_legs(subscript)
    shared = set(legs[0] + legs[1])
    out    = legs[2]

    symlabels_C = []
    for lg in out:

        if   lg in legs[0]:
             iA = legs[0].find(lg)
             s  = symlabels_A[iA] 
        else:
             iB = legs[1].find(lg)
             s  = symlabels_B[iB]

        symlabels_C.append(s)

    return symlabels_C
    



def compute_output_fullsigns(subscript, fullsigns_A, fullsigns_B):

    # Compute output fullsigns
    legs   = subscript_to_legs(subscript)
    shared = set(legs[0] + legs[1])
    out    = legs[2]

    phase = compute_phase(subscript, signs_A, signs_B) 

    fullsigns_C = []
    for lg in out:

        if   lg in legs[0]:
             iA = legs[0].find(lg)
             s  = fullsigns_A[iA] * phase
        else:
             iB = legs[1].find(lg)
             s  = fullsigns_B[iB]

        fullsigns_C.append(s)

    return symlabels_C













































































































































































































































































































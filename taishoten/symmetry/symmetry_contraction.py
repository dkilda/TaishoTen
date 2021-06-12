#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

import numpy as np
import copy  as cp
import itertools

import taishoten.util as util

from taishoten.util import Str, dictriplet
from taishoten.util import zip_shared, zip_compress, get_shared_indices
from taishoten.util import assertequal, assertclose, isiterable
from taishoten.util import IS, ISNOT, ARE, ARENOT

from .map import Map, contract_maps



# --- Symmetry contraction class -------------------------------------------- #

def compute_symmetry_contraction(symA, symB, legs):

    symcon = SymmetryContraction(symA, symB, legs)
    symcon.compute()
    return symcon



class SymmetryContraction:

   def __init__(self, symA, symB, legs):

       # Sanity checks on symA and symB
       msg = "SymmetryContraction: symA and symB must have equal dimensions"
       assertequal(symA.ndim, symB.ndim, msg)

       msg = "SymmetryContraction: symA and symB types must be the same"
       assertequal(type(symA), type(symB), msg)

       msg = "SymmetryContraction: symA and symB must have equal moduli"
       if  IS(symA.mod) and IS(symB.mod):
           assertclose(symA.mod, symB.mod, msg)

       # Input symmetries and legs      
       self._sym  = dictriplet(symA, symB, None)
       self._legs = legs

       # Signed legs of A and B
       self._symlegs = dictriplet(None, None, None)
       self._make_symlegs("A")
       self._make_symlegs("B")

       # Compute relative phase between A and B leg signs/directions
       # (phase = 1 if directions are compatible, -1 if they are not) 
       self._phase = self._compute_relative_phase()



   # --- Initialization ----------------------------------------------------- #

   def _make_symlegs(self, key, fullsigns=None): 

       # Default fullsigns
       if  ISNOT(fullsigns):
           fullsigns = self.sym(key).fullsigns

       # Make signed symmetric legs corresponding to A, B or C tensor
       self._symlegs[key] = util.make_symlegs(self.legs(key), fullsigns)
       return self



   def _compute_relative_phase(self):

       # Legs and their signs for A and B tensors
       signsA = (self.symlegs("A"), self.sym("A").signs)
       signsB = (self.symlegs("B"), self.sym("B").signs)

       # Determine phases of all shared legs.
       # -- leg on A and B has opposite (compatible) sign: phase=1
       # -- leg on A and B has same   (incompatible) sign: phase=-1
       phases = [(-1 if sA == sB else 1) \
                     for sA, sB in zip_shared(signsA, signsB)]

       # No shared legs: phase is trivially 1
       if  len(phases) == 0:
           phase = 1
           return phase

       # All shared legs must have the same phase (one unique phase)
       # else it means the symmetry contraction is not valid
       msg = "SymmetryContraction.compute_relative_phase:"\
             "all shared legs must have the same phase"
       assertequal(np.unique(phases).size, 1, msg)

       # Return the relative phase
       phase = phases[0]
       return phase



   # --- Getters and properties --------------------------------------------- #
   
   def sym(self, key=None):
       if  ISNOT(key):
           return self._sym
       return self._sym[key]


   def legs(self, key=None):
       if  ISNOT(key):
           return self._legs
       return self._legs[key]


   def symlegs(self, key=None):
       if  ISNOT(key):
           return self._symlegs
       return self._symlegs[key]


   @property
   def phase(self):
       return self._phase


   @property 
   def ndim(self):
       return self.sym("A").ndim


   @property
   def type(self):
       return type(self.sym("A"))



   # --- Compute the output symmetry after contraction ---------------------- #

   def compute(self):

       # Compute output mod, qtot, full signs, and symlabels for symC
       mod       = self.output_mod()
       qtot      = self.output_qtot()
       fullsigns = self.output_fullsigns()
       symlabels = self.output_symlabels(fullsigns)

       # Compute and return the output symmetry symC
       self._sym["C"] = self.type(fullsigns, symlabels, qtot, mod)
       return self



   def output_qtot(self):

       # Compute the total output symlabel, Qtot
       return (self.phase * self.sym("A").qtot) + self.sym("B").qtot     



   def output_mod(self):

       # Determine the output modulus (if both moduli are present,
       # we have already asserted they are equal in __init__)
       modA = self.sym("A").mod
       modB = self.sym("B").mod

       if   ARENOT(modA, modB):
            return None

       elif ISNOT(modA) and IS(modB):
            return modB

       elif IS(modA) and ISNOT(modB):
            return modA

       elif ARE(modA, modB):
            return modA

       else:
            msg = "SymmetryContraction.output_mod: " \
                  "invalid modA = {} or modB = {}".format(modA, modB)
            raise NotImplementedError(msg)



   def output_symlabels(self, fullsigns):

       # Make symmetric legs for C using the new full signs
       self._make_symlegs("C", fullsigns)

       # Legs and their symlabels for A and B, legs of C
       symlabels_A = (self.symlegs("A"), self.sym("A").symlabels)
       symlabels_B = (self.symlegs("B"), self.sym("B").symlabels)
       legs_C      =  self.symlegs("C")

       # Determine symlabels for all symlegs of C
       symlabels_C = zip_compress(symlabels_A, symlabels_B, legs_C)
       symlabels_C = list(symlabels_C)
       return symlabels_C



   def output_fullsigns(self):

       # Multiply full signs of A by the relative phase, 
       # to make A and B leg directions compatible
       fullsigns_A = util.phased_signs(self.sym("A").fullsigns, self.phase)   
       fullsigns_B =                   self.sym("B").fullsigns

       # Legs and their full signs for A and B, legs of C
       fullsigns_A = (self.legs("A"), fullsigns_A)
       fullsigns_B = (self.legs("B"), fullsigns_B)
       legs_C      =  self.legs("C")

       # Determine full signs for all symlegs of C
       fullsigns_C = zip_compress(fullsigns_A, fullsigns_B, legs_C)
       fullsigns_C = ''.join(fullsigns_C)
       return fullsigns_C



   # --- Alignment of symlabels --------------------------------------------- #

   def align_symlabels(self, symlabels_A, symlabels_B): 

       # Shortcut
       symA = self.sym("A")
       symB = self.sym("B")
       modA = symA.mod
       modB = symB.mod

       # Use internal align method from either of the symmetries
       _align = symA.align_symlabels

       if   ARE(modA, modB) or ARENOT(modA, modB):

            # Align symlabels of A and B directly 
            # (no moduli or moduli are equal, which is asserted in __init__)
            return _align(symlabels_A, symlabels_B) 

       elif ISNOT(modA) and IS(modB):

            # Fold A using modulus of B
            # (to put A on the same footing as B, so we can align them)
            folded_symlabels_A = symB.fold(symlabels_A)
            return _align(symlabels_B, folded_symlabels_A)

       elif IS(modA) and ISNOT(modB):
            # Fold B using modulus of A
            # (to put B on the same footing as A, so we can align them)
            folded_symlabels_B = symA.fold(symlabels_B)
            return _align(symlabels_A, folded_symlabels_B)

       else:
            msg = "SymmetryContraction.align_symlabels: " \
                  "invalid modA = {} or modB = {}".format(modA, modB)
            raise NotImplementedError(msg)
            



# --- Compute the auxiliary symmetry and maps ------------------------------- #

def compute_aux_symmetry(symcon):

    # Shortcut
    symA  = symcon.sym("A")
    symB  = symcon.sym("B")
    legsA = symcon.symlegs("A")
    legsB = symcon.symlegs("B")

    # Get indices of shared symlegs on A and B
    shared_idx_A, shared_idx_B = util.get_shared_indices(legsA, legsB)

    # The symlabels B have eff_phase = -phase wrt symlabels A:  
    # because same shared leg has signA = -s * phase, signB = s
    phase = -1 * symcon.phase

    # Compute auxiliary symlabels for A and B, align them to obtain a minimal
    # set of A symlabels that satisfy the conservation laws at A and B nodes
    aux_symlabels_A = symA.aux_symlabels(shared_idx_A) 
    aux_symlabels_B = symB.aux_symlabels(shared_idx_B, phase=phase)
    aux_symlabels   = symcon.align_symlabels(aux_symlabels_A, aux_symlabels_B) 
    
    # We will construct an auxiliary tensor/map whose symlegs consist of
    # the shared symlegs from A and the new auxiliary leg. 
    signs = \
         ''.join(symA.signs[i]     for i in shared_idx_A) + '-'
    symlabels = [symA.symlabels[i] for i in shared_idx_A] + [aux_symlabels]
    
    mod = symA.mod if IS(symA.mod) else symB.mod

    # Construct the auxiliary symmetry defining an auxiliary tensor/map
    aux_sym = symcon.type(signs, symlabels, 0, mod)
    return aux_sym



def compute_maps(symcon, backend=None):  

    """
    E.g. if we have maps of A and B with legs [IJLM, NOJL], we will create 
    an additional auxiliary map with legs JLQ, so that now we have three maps
    with legs [IJLM, NOJL, JLQ]. Next, we compute all possible pairwise 
    contractions of these maps, to obtain a full map list.

    """

    # Shortcut function for creating new maps and adding them to the list
    maps = []
    def add_map(*args, **kwargs):
        mp = Map.compute(*args, backend=backend, **kwargs)
        maps.append(mp)

    # Shortcut vars
    symA  = symcon.sym("A")
    symB  = symcon.sym("B")
    legsA = symcon.symlegs("A")
    legsB = symcon.symlegs("B")

    # Compute irrep maps representing the symmetries of A and B tensors
    add_map(symA, legsA) 
    add_map(symB, legsB)

    # Get shared and extra symlegs from symmetry contractor
    sharedAB = legsA & legsB 
    extraA   = legsA - sharedAB 
    extraB   = legsB - sharedAB 

    # If the numbers of shared and extra legs are more than one, we will get 
    # output tensor C with two hidden legs --> i.e. we cannot get 
    # a valid output representable by a node with just one hidden symleg.
    #
    # Solution: irrep alignment algorithm, where we transform maps of A and B 
    # into reduced maps (containing A and B extra legs and auxiliary leg "Q")
    # and auxiliary map (containing shared legs        and auxiliary leg "Q"). 
    # This way, we can also express the output node with two hidden legs as a 
    # composite structure of two nodes: reduced map and auxiliary map, 
    # each containing only one hidden leg.
    #
    # First, we need to compute the auxiliary map.
    if  len(sharedAB) > 1 and len(extraA) > 1 and len(extraB) > 1:

        # Compute auxiliary symmetry and the corresponding symlegs
        aux_sym     = compute_aux_symmetry(symcon)
        aux_maplegs = sharedAB + Str("Q")

        # Create auxiliary map representing the auxiliary symmetry
        add_map(aux_sym, aux_maplegs) 
        
    # Calculate all pairwise contractions of A, B and auxiliary maps: gives 
    # a complete list of maps required for all possible symleg transformations
    maps = compute_pairwise_contractions_of_maps(maps)
    return maps




def compute_pairwise_contractions_of_maps(maps):

    # Get legs of initial maps
    maplegs = util.get_legs(maps)

    # Loop over all possible pairings of maps, 
    # we will contract each valid pair to produce a new map
    for mapI, mapJ in util.combinations(maps, 2):

        # For a given pair of maps = (mapI, mapJ)
        # find all, shared, and output legs
        ALL    = mapI.legs | mapJ.legs
        SHARED = mapI.legs & mapJ.legs
        OUT    = ALL - SHARED 

        # No shared legs: cannot contract mapI and mapJ. Skip.
        if  len(SHARED) == 0:
            continue

        # No output legs: cannot produce a valid map that has legs.
        # Output legs correspond to one of existing maps: 
        # the output map is already on our list. Skip.
        if  len(OUT) == 0 or (OUT in maplegs):
            continue

        # Contract mapI and mapJ --> gives a new map
        new_map = contract_maps(mapI, mapJ, OUT) 

        # Add the new map and its legs to the map list
        maps.append(new_map)
        maplegs.append(OUT)

    return maps













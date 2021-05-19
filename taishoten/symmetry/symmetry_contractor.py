#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

import numpy as np
import copy  as cp
import itertools

import taishoten.util as util

from taishoten.util import StrSet
from taishoten.util import assertequal, assertclose, isiterable
from taishoten.util import IS, ISNOT, ARE, ARENOT

from taishoten.params import ALPHABET, SIGN, FLIP, SYMTOL

from map import Map, contract_maps




class SymmetryContractor: 

   def __init__(self, symA, symB, denselegs):

       # Sanity check: symA and symB must have equal dimensions
       msg = "SymmetryContractor: symA and symB must have equal dimensions"
       assertequal(symA.ndim, symB.ndim, msg)

       # Sanity check: symA and symB must have the same type
       msg = "SymmetryContractor: symA and symB types must be the same"
       assertequal(type(symA), type(symB), msg)

       # Sanity check: symA and symB must have consistent moduli
       msg = "SymmetryContractor: symA and symB must have equal moduli"
       if  IS(symA.mod) and IS(symB.mod):
           assertclose(symA.mod, symB.mod, msg)

       # Input symmetries and denselegs      
       self._sym       = [symA, symB, None]
       self._denselegs = denselegs

       # Make symlegs
       self._symlegs = [None] * len(self._sym)
       self.make_symlegs("A")
       self.make_symlegs("B")

       # Get the relative phase between A and B leg signs/directions
       # (phase = 1 if directions are complementary, -1
       self._phase = self.compute_relative_phase()



   # --- Initialization ----------------------------------------------------- #

   def make_symlegs(self, key, fullsigns=None):

       # Get idx corresponding to the key
       i = self._key_to_idx(key)

       # Default fullsigns
       if  ISNOT(fullsigns):
           fullsigns = self.fullsigns(i)

       # Make symmetric legs corresponding to A, B or C tensor
       self._symlegs[i] = util.make_symlegs(self.denselegs(i), fullsigns)
       return self



   def compute_relative_phase(self):

       # Shortcut
       signs_AB          = self.signs("AB")
       symlegs_AB        = self.symlegs("AB")
       shared_symlegs_AB = self.symlegs("A") & self.symlegs("B") 

       # If no shared legs, the phase is trivially 1
       if  len(shared_symlegs) == 0:
           phase = 1
           return phase

       # Signs of all shared legs: yields (sign from symA, sign from symB)
       gen_signs    = util.gen_binary_data("AND")
       gen_signs_AB = gen_signs(shared_symlegs_AB, symlegs_AB, signs_AB)

       # Determine the phases of all shared legs. 
       # If both shared legs have opposite signs (compatible directions): 
       # phase = 1, if equal signs (incompatible directions): phase = -1. 
       phase_list = [(-1 if sgnA == sgnB else 1) \
                         for sgnA, sgnB in gen_signs_AB]

       # All shared legs must have the same phase (only one unique element), 
       # else it means the symmetry is invalid 
       num_unique_phases = np.unique(phase_list).size
       assertequal(num_unique_phases, 1, \
                   "SymmetryContractor._compute_relative_phase:"\ 
                   "all shared symlegs must have the same phase")

       # Determine the relative phase
       phase = phase_list[0]
       return phase



   # --- Internal, auxiliary get methods ------------------------------------ #

   def _get(self, x, which=None):

       # Default selector
       which = which if IS(which) else "ALL"

       # Return selected legs or data
       if   which == "ALL":
            return tuple(x)

       elif which in ("AB", "IN"):
            return tuple(x)[:-1]

       elif which in ("C", "OUT"):
            return tuple(x)[-1]

       elif which in ("A", "B", "C"):
            return x[self._key_to_idx(which)]

       elif which in (0, 1, 2):
            return x[which]

       else:
            msg = "SymmetryContractor._get: invalid which, {}"
            msg = msg.format(which)
            raise ValueError(msg)



   def _get_sym_attr(self, attr, which=None):

       # Return a given attribute of symmetry/symmetries
       sym = self.sym(which)

       if   isiterable(sym):
            return (getattr(s,   attr) for s in sym)
       else:
            return  getattr(sym, attr)


   def _key_to_idx(self, key):
       return {"A": 0, "B": 1, "C": 2}[key]

   

   # --- Getters and properties --------------------------------------------- #

   @property
   def phase(self):
       return self._phase

   @property 
   def ndim(self):
       return self.sym("A").ndim

   @property
   def symtype(self):
       return type(self.sym("A"))


   def denselegs(self, which=None):
       return self._get(self._denselegs, which=which)

   def symlegs(self, which=None):
       return self._get(self._symlegs,   which=which)


   def sym(self, which=None):
       return self._get(self._sym, which=which) 

   def symlabels(self, which=None):
       return self._get_sym_attr("symlabels", which=which)

   def signs(self, which=None):
       return self._get_sym_attr("signs", which=which)

   def fullsigns(self, which=None):
       return self._get_sym_attr("fullsigns", which=which)

   def mod(self, which=None):
       return self._get_sym_attr("mod", which=which)

   def qtot(self, which=None):
       return self._get_sym_attr("qtot", which=which)



   # --- Compute the output symmetry after contraction ---------------------- #

   def compute_output_sym(self):

       # Compute the output mod, qtot and full-signs for symC
       mod       = self.output_mod()
       qtot      = self.output_qtot()
       fullsigns = self.output_fullsigns()

       # Make symmetric legs for C using the new full-signs
       self.make_symlegs("C", fullsigns)

       # Compute output symlabels for symC
       symlabels = self.output_symlabels()

       # Compute and return the output symmetry symC
       self._symC = self.symtype(fullsigns, symlabels, qtot, mod)
       return self._symC



   def output_qtot(self):

       # Compute the total output symlabel, Qtot
       return (self.phase * self.qtot("A")) + self.qtot("B")     



   def output_mod(self):

       # Determine the output modulus (if both moduli are present,
       # we have already asserted they are equal in __init__)
       modA = self.mod("A")
       modB = self.mod("B")

       if   ISNOT(modA) and ISNOT(modB):
            return None

       elif ISNOT(modA) and IS(modB):
            return modB

       elif ISNOT(modB) and IS(modA):
            return modA

       else:
            return modA



   def output_symlabels(self):

       # Shortcut
       symlegs_C    = self.symlegs("C")
       symlegs_AB   = self.symlegs("AB")
       symlabels_AB = self.symlabels("AB") 

       # Shortcut to generator 
       gen_symlabels = util.gen_binary_data("OR")

       # Determine symlabels for all symlegs of C.
       # If a given symleg of C is in A: get its symlabels from A,
       # Else:                           get its symlabels from B.
       # NB every symleg of C must come from either A or B.
       symlabels_C = gen_symlabels(symlegs_C, symlegs_AB, symlabels_AB)
       symlabels_C = np.array(list(symlabels_C))
       return symlabels_C



   def output_fullsigns(self):

       # Function to get the phased full-signs. 
       # If phase = -1: flip the full-signs of all A legs OR all B legs, 
       #                to make A and B leg directions compatible.
       # If phase =  1: A and B leg directions are already compatible.
       def phased_fullsigns(k):
           if   self.phase == 1:
                return self.fullsigns(k)
           else: 
                return self.sym(k).flipped_fullsigns
         
       # Shortcut (full-signs concern all legs, both those that have symmetry 
       # and those that do not, so we will loop over dense legs).
       # Get full-signs of A and B legs, accounting for their relative phase.
       denselegs_C  = self.denselegs("C")
       denselegs_AB = self.denselegs("AB")
       fullsigns_AB = [self.phased_fullsigns("A"), self.fullsigns("B")]

       # Shortcut to generator 
       gen_fullsigns = util.gen_binary_data("OR")

       # Determine full-signs for all legs of C.
       # If a given leg of C is in A: get its sign from A,
       # Else:                        get its sign from B.
       fullsigns_C   = gen_fullsigns(denselegs_C, denselegs_AB, fullsigns_AB)
       fullsigns_C   = ''.join(fullsigns_C)
       return fullsigns_C



   # --- Compute the auxiliary symmetry for irrep alignment algorithm ------- #

   def compute_aux_sym(self):

       # Shortcut
       symA = self.sym("A")
       symB = self.sym("B")
       symlegs_A = self.symlegs("A")
       symlegs_B = self.symlegs("B")
       shared_symlegs_AB = symlegs_A & symlegs_B 

       # Get idx of shared symlegs in A and B
       shared_idx_A = util.gen_idx(shared_symlegs_AB, symlegs_A) 
       shared_idx_B = util.gen_idx(shared_symlegs_AB, symlegs_B)

       # The same shared leg has sign =  s * phase when going to/from A
       #                     and sign = -s         when going to/from B.
       # --> so symlabels  of the shared leg on B have eff_phase = -phase
       #     wrt symlabels of the shared leg on A. 
       phase = -1 * self.phase

       # Compute auxiliary symlabels for A and B tensors, align them to 
       # make a minimal set of A symlabels that satisfy the conservation 
       # laws at A and B tensor nodes.
       aux_symlabels_A = symA.aux_symlabels(shared_idx_A) 
       aux_symlabels_B = symB.aux_symlabels(shared_idx_B, phase=phase)
       aux_symlabels = self.align_symlabels(aux_symlabels_A, aux_symlabels_B)

       # We will construct an auxiliary tensor/map whose symlegs consist of 
       # the shared symlegs from A and the new auxiliary leg. 
       # So this auxiliary tensor/map will have:
       # Signs     = signs     of shared A symlegs + '-ve' sign of aux symleg
       # Symlabels = symlabels of shared A symlegs + aux_symlabels
       signs      = ''.join(symA.signs[i]     for i in shared_idx_A) 
       symlabels  =        [symA.symlabels[i] for i in shared_idx_A]

       signs     += '-'
       symlabels += [aux_symlabels]

       # Construct the auxiliary symmetry defining an auxiliary tensor/map.
       aux_sym = self.symtype(signs, symlabels)
       return aux_sym       



   def align_symlabels(self, symlabels_A, symlabels_B):

       # Shortcut
       modA = self.mod("A")
       modB = self.mod("B")

       # Use internal align method from either of the symmetries
       _align = self.sym("A").align_symlabels

       if   ARE(modA, modB) or ARENOT(modA, modB):

            # Align symlabels of A and B directly 
            # (no moduli or moduli are equal, which is asserted in __init__)
            return _align(symlabels_A, symlabels_B) 

       elif ISNOT(modA) and IS(modB):

            # Fold A using modulus of B
            # (to put A on the same footing as B, so we can align them)
            folded_symlabels_A = self.sym("A").fold(symlabels_A)
            return _align(symlabels_B, folded_symlabels_A)

       else:
            # Fold B using modulus of A
            # (to put B on the same footing as A, so we can align them)
            folded_symlabels_B = self.sym("B").fold(symlabels_B)
            return _align(symlabels_A, folded_symlabels_B)
            

   # ------------------------------------------------------------------------ #




def compute_maps(symcon, backend):  

    """
    E.g. if we have maps of A and B with legs [IJLM, NOJL], we will create 
    an additional auxiliary map with legs JLQ, so that now we have three maps
    with legs [IJLM, NOJL, JLQ]. Next, we compute all possible pairwise 
    contractions of these maps, to obtain a full map list.

    """

    # Shortcut function for creating new maps and adding them to the list
    maps = []
    def add_map(*args, **kwargs):
        new_map = Map.compute(*args, backend=backend, **kwargs)
        maps.append(new_map)

    # Shortcut vars
    symA = symcon.sym("A")
    symB = symcon.sym("B")
    symlegs_A  = symcon.symlegs("A")
    symlegs_B  = symcon.symlegs("B")
    symlegs_AB = symcon.symlegs("AB")

    # Compute irrep maps representing the symmetries of A and B tensors
    add_map(symA, symlegs_A) 
    add_map(symB, symlegs_B)

    # Get shared and extra symlegs from symmetry contractor
    shared_AB = symlegs_A & symlegs_B 
    extra_A   = symlegs_A - shared_AB 
    extra_B   = symlegs_B - shared_AB 

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
    if  len(shared_AB) > 1 and len(extra_A) > 1 and len(extra_B) > 1:

        # Compute auxiliary symmetry and the corresponding symlegs
        aux_sym     = symcon.compute_aux_sym()
        aux_maplegs = shared + StrSet("Q")

        # Create auxiliary map representing the auxiliary symmetry
        add_map(aux_sym, aux_maplegs) 

    # Calculate all pairwise contractions of A, B and auxiliary maps: gives 
    # a complete list of maps required for all possible symleg transformations
    maps = compute_pairwise_contractions_of_maps(maps)
    return maps




def compute_pairwise_contractions_of_maps(maps):

    # Get legs of initial maps
    maplegs = get_maplegs(maps)

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
        # Output legs correspond to one of existing maps: the output 
        # map is already on our list. Skip.
        if  len(OUT) == 0 or (OUT in maplegs):
            continue

        # Contract mapI and mapJ --> gives a new map
        subscript = util.legs_to_subscript(mapI.legs, mapJ.legs, OUT)
        new_map   = contract_maps(subscript, mapI, mapJ) 

        # Add the new map and its legs to the map list
        maps.append(new_map)
        maplegs.append(OUT)

    # Sort maps by the length of their map legs
    sorted_maplegs = sorted(maplegs, key=len)
    sorted_maps    = list(util.gen_data(sorted_maplegs, maplegs, maps))
    return sorted_maps




def get_maplegs(maps):
    return tuple([mp.legs for mp in maps])

















































































































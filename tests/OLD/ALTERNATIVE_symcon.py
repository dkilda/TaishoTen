#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

TODO



Tests:

-- functions for brute-force calculation of flat symlabels, aux symlabels and aligned symlabels

-- assertSymmetry, assertSymmetryContractor, assertMap, assertTransformGraph, assertTensor methods

-- test_symeinsum: wrap the testing steps of each symeinsum in a function



SymmetryContractor:

--? denselegs: input subscript instead, convert it to a list of legs inside SymmetryContractor.

-- ext functions phased_signs flipped_signs, where phased_signs multiplies signs by phase factor 
   and flipped_signs is a special case of flipped_signs with phase=-1.

-- preprocess signs input: input either a string '++0-' or a tuple of ints (1,1,0,-1), always convert it
   to a tuple of ints (for easier sign algebra)?

-- within SymmetryContractor, abolish the concept of symlegs and denselegs. Just call them legs instead.
   Might have to differentiate b/n signed and unsigned legs (using signed/unsigned idx methods of Symmetry class).

-- Three basic generators: 
   zip_select/zip_compress(items_lst, legs_lst, legs_selectors) (zip symlabels_A,B and select only items that are in some select list provided as input),
   zip_common/zip_shared(items_lst, legs_lst)                (we zip signs_A,B     and take only common elements, i.e. do shared inside) 
   get_indices(lst, items), generalization of lst.index(item) --> perhaps make this return list instead of being a generator?

   get(legs, items, l): shortcut for: i = legs.find(l); s = items[i].

   What about: dA = dict(zip(slegs("A"), symlabels("A"))), dB = dict(zip(slegs("B"), symlabels("B")))? Then we can do dA[sleg] -> symlabels for this sleg.
   (sleg = signed leg, leg = any leg). Use function symlabels_A = get_dict(self.slegs("A"), self.sym("A").symlabels).

-- Accessing legs in SymmetryContractor: self.legs("A"), self.legs("B"). Same for accessing symmetries (note we use func not dict access).

-- Shortcuts: only use legsK = self.legs(key), symK = self.sym(key).

-- SymmetryContractor class should do only one thing: the single responsibility principle. 

   SymCon has three things in common: (symA, symB, symC), phase, legs.
   It should only construct symC from symA, symB and store phase computed in the process.
   
   compute_aux_sym() should take symcon as input
   make_symlegs() should also take symcon as input

   rename SymmetryContractor -> ContractedSymmetry/SymmetryContraction?



Symmetry:

--? @property methods that return signed idx and unsigned idx.

-- remove @property methods for num_signed_legs and num_unsigned_legs

-- indices input to aux_symlabels and flatten_symlabels: 
   convert indices input from all-legs indices to signed-legs indices. Should simplify things a lot!

--? similarly, make an ext func that converts b/n fullsigns and signs

--? instead of keeping SIGNS and FLIPPED global variables, wrap them in functions like int_signs(signs):, flipped_signs(signs), etc.
   Same with ALPHABET: wrap it in make_legs() func

-- aux_symlabels: rename to lhs/rhs_symlabels? But mid/edge might be clearer as to which symlabels we're keeping on LHS/RHS...

-- summed meshgrid and ravel external functions for merging symlabels and/or flattening symlabels



util:

-- sym_dense_legs_to_subscript(symlegs, denselegs) and make_full_subscript(symlegs, subscript)




"""














































































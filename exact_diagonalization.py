#!/usr/bin/env python
# coding: utf-8

import numpy, sys
from inspect import getmembers, isfunction
import config
import initialization
import potentials

#print(getmembers(initialization,isfunction))
config.Alpha = 1.
initialization.CreateLattice( nx=2, d=2, z=6 )
initialization.CreateEwald()
initialization.CreateBasis( config.ns / 2 )
print(config.Basis.shape)

Disorder = 2.
H_Disorder = potentials.CreateDisorderHam( Disorder )
print( type( H_Disorder ) )

V = 1.
H_NN = potentials.CreateNearestNeighborHam( V )
print( type( H_NN ) )

H_LR = potentials.CreateLongRangeHam( V )
print( type( H_LR ) )

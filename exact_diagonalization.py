#!/usr/bin/env python
# coding: utf-8

import numpy, sys
from inspect import getmembers, isfunction
import config
import initialization
import potentials
import kinetic_flux
import diagonalization

#print(getmembers(initialization,isfunction))
config.Alpha = 1.
initialization.CreateLattice( nx=2, d=2, z=6 )
initialization.CreateEwald()
initialization.CreateBasis( 1 )
print(config.Basis.shape)

Disorder = 2.
H_Disorder = Disorder * potentials.CreateDisorderHam()

H_NN = potentials.CreateNearestNeighborHam()

H_LR = potentials.CreateLongRangeHam()

Kin = 1.
Flux = numpy.array( [ 0.34 , 0.82 ] )
H_Kin = kinetic_flux.CreateKineticHam( Kin, Flux )

V = 0.
E0 = diagonalization.Diagonalization( H_Kin + V * H_LR, 'Brute' )
print(E0)

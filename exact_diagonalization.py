#!/usr/bin/env python
# coding: utf-8

import numpy, sys
from inspect import getmembers, isfunction
import config
import initialization
import potentials
import kinetic_flux
import diagonalization
import observables

# set values of the different parameters
Kin = 1.
V = 25.
Disorder = 0.
config.Alpha = 1.
NFluxes = 4
config.udir = 0.
config.vdir = 0.

# initialization
initialization.CreateLattice( nx=2, d=2, z=6 )
initialization.CreatePhiMatrix()
initialization.CreateEwald()
initialization.CreateBasis( config.ns // 2 )
initialization.CreateSuffix( V, NFluxes )
H_LR = potentials.CreateLongRangeHam()

# initialize histograms
nPhis = 20000
PofPhi = numpy.zeros( ( nPhis, 2 ) )
PofPhi[ :, 0 ] = numpy.linspace( -10., 10., num=nPhis, endpoint=False )

nOmegas = 10000
CSF = numpy.zeros( ( nOmegas, config.ns + 1 ) )
CSF[ :, 0 ] = numpy.linspace( 0., 50., num=nOmegas, endpoint=False )

D = numpy.zeros( ( 3, ) ) # Drude, Fxx, Jx

nOmegas = 10000
OptCond = numpy.zeros( ( nOmegas, 2 ) )
OptCond[ :, 0 ] = numpy.linspace( 0., 50., num=nOmegas, endpoint=False )

DensCorrK = numpy.zeros( ( config.ns, ) )

# loop over the fluxes and average the observables
if config.Dimensions == 1 :
    for phix in numpy.linspace( 0, 1, num=NFluxes, endpoint=False ) :
        # set up kinetic portion of Hamiltonian
        Flux = numpy.array( [ phix ] )
        H_Kin = kinetic_flux.CreateKineticHam( Kin, Flux )

        # diagonalize Hamiltonian
        E0, Psi0, Evals, Evecs = diagonalization.Diagonalization( H_Kin + V * H_LR, 'Lanczos' )

        # compute observables
        PofPhi[ :, 1 ] += observables.ComputePofPhi( Psi0, PofPhi )
        CSF[ :, 1: ] += observables.ComputeChargeSpectralFunction( Psi0, CSF, Evecs, Evals )
        OptCond[ :, 1 ] += observables.ComputeOptCond( Psi0, Evecs, Evals, H_Kin, OptCond )
        DensCorrK += observables.ComputeDensityCorrelationKSpace( Psi0 )
else :
    for phix in numpy.linspace( 0, 1, num=NFluxes, endpoint=False ) :
        for phiy in numpy.linspace( 0, 1, num=NFluxes, endpoint=False ) :
            # set up kinetic portion of Hamiltonian
            Flux = numpy.array( [ phix, phiy ] )
            H_Kin = kinetic_flux.CreateKineticHam( Kin, Flux )

            # diagonalize Hamiltonian
            E0, Psi0, Evals, Evecs = diagonalization.Diagonalization( H_Kin + V * H_LR, 'Lanczos' )

            # compute observables
            PofPhi[ :, 1 ] += observables.ComputePofPhi( Psi0, PofPhi )
            CSF[ :, 1: ] += observables.ComputeChargeSpectralFunction( Psi0, CSF, Evecs, Evals )
            D += observables.ComputeDrudeWeight( Psi0, Evecs, Evals, Kin, Flux )
            OptCond[ :, 1 ] += observables.ComputeOptCond( Psi0, Evecs, Evals, H_Kin, OptCond )
            DensCorrK += observables.ComputeDensityCorrelationKSpace( Psi0 )

# Average over NFluxes
if config.Dimensions == 1 :
    PofPhi[ :, 1 ] = PofPhi[ :, 1 ] / NFluxes
    CSF[ :, 1: ] = CSF[ :, 1: ] / NFluxes
    D = D / NFluxes
    DensCorrK = DensCorrK / NFluxes
else :
    PofPhi[ :, 1 ] = PofPhi[ :, 1 ] / ( NFluxes * NFluxes )
    CSF[ :, 1: ] = CSF[ :, 1: ] / ( NFluxes * NFluxes )
    D = D / ( NFluxes * NFluxes )
    OptCond[ :, 1 ] = OptCond[ :, 1 ] / ( NFluxes * NFluxes )
    DensCorrK = DensCorrK / ( NFluxes * NFluxes )

print(D)
print(DensCorrK)
print(numpy.sum(DensCorrK))

# write results
numpy.savetxt( "PofPhi" + config.Suffix, PofPhi )
numpy.savetxt( "CSF" + config.Suffix, CSF )
numpy.savetxt( "Drude" + config.Suffix, D )
numpy.savetxt( "OptCond" + config.Suffix, OptCond )
numpy.savetxt( "DensCorrK" + config.Suffix, DensCorrK )

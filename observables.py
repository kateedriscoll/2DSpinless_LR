import config
import numpy, sys
from scipy.sparse import coo_matrix

# compute PofPhi
def ComputePofPhi( Psi0, PofPhi ) :
    for si, state in enumerate( config.Basis ) :
        n = numpy.copy( state )
        n = n - ( config.np / config.ns )
        # compute the phis for all of the sites
        phis = numpy.asarray( [ numpy.sum( config.MatrixDist[ i, : ] * n ) for i in range( config.ns ) ] )

        # find the indices that these phis correspond to in the histogram
        inds = numpy.digitize( phis , PofPhi[ : , 0 ] )

        # count the frequency of these indices and add the non-zero frequencies to the histogram
        freq = numpy.bincount( inds )
        ii = numpy.nonzero( freq )[ 0 ]
        PofPhi[ ii, 1 ] += freq[ ii ] * numpy.linalg.norm( Psi0[ si ] )**2.0

    return PofPhi[ :, 1 ]

# compute the spectral function of charge fluctuations
def ComputeChargeSpectralFunction( Psi0, CSF, Evecs, Evals ) :
    VecCqw = numpy.zeros( ( len( config.Basis ), config.ns ), dtype='complex128' )

    for si, s in enumerate( config.Basis ) :
        for j in range( config.ns ) :
            Op = 0.0
            for i in range( config.ns ) :
                Op += ( s[ i ] - config.np / config.ns ) * config.PhiMatrix[ j, i ] / numpy.sqrt( config.ns )
            VecCqw[ si, j ] += Op * Psi0[ si ]

    for n in range( Evecs.shape[ 1 ] ) :
        Omega_n = Evals[ n ] - Evals[ 0 ]
        ind = numpy.digitize( numpy.asarray( [ Omega_n ] ), CSF[ :, 0 ] )
        if ind == len( CSF[ :, 0 ] ) :
            ind = ind-1
        for q in range( config.ns ) :
            VecCqw_q = VecCqw[ :, q ]
            Result = numpy.linalg.norm( Evecs[ :, n ].transpose() * VecCqw_q )
            CSF[ ind, q+1 ] += Result

    return CSF[ :, 1: ]

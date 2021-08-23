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

import config
import numpy, sys
from scipy.sparse import coo_matrix

# create Hamiltonian of disordered potential for given basis
def CreateDisorderHam() :
    row = numpy.array( range( len( config.Basis ) ) )
    col = numpy.array( range( len( config.Basis ) ) )
    data = numpy.zeros( ( len( config.Basis ), ) )

    SiteDisorders = numpy.random.uniform( low=-1, high=1, size=( config.ns, ) )

    for statei, state in enumerate( config.Basis ) :
        Potential = numpy.dot( state, SiteDisorders )
        data[ statei ] +=  Potential

    # initialize the sparse matrix with non-zero data stored in the ( row, col ) entries
    H_Disorder = coo_matrix( ( data, ( row, col ) ) , shape=( len( config.Basis ), len( config.Basis ) ) )
    return H_Disorder

# create Hamiltonian of nearest-neighbor potential for given basis
def CreateNearestNeighborHam() :
    row = numpy.array( range( len( config.Basis ) ) )
    col = numpy.array( range( len( config.Basis ) ) )
    data = numpy.zeros( ( len( config.Basis ), ) )

    for statei, state in enumerate( config.Basis ) :
        Potential = 0.0
        for s in range( len( state ) ) :
            for n in range( config.Connectivity ) :
                Potential += 0.5 * state[ s ] * state[ config.Neighbors[ s, n ] ]
        data[ statei ] +=  Potential

    # initialize the sparse matrix with non-zero data stored in the ( row, col ) entries
    H_NearestNeighbors = coo_matrix( ( data, ( row, col ) ) , shape=( len( config.Basis ), len( config.Basis ) ) )
    return H_NearestNeighbors

# create Hamiltonian of long-range potential for given basis
def CreateLongRangeHam() :
    row = numpy.array( range( len( config.Basis ) ) )
    col = numpy.array( range( len( config.Basis ) ) )
    data = numpy.zeros( ( len( config.Basis ), ) )

    for statei, state in enumerate( config.Basis ) :
        n = numpy.copy( state )
        n = n - ( config.np / config.ns )
        ninj = n[ numpy.newaxis , : ].T @ n[ numpy.newaxis , : ] - numpy.diag( n * n )
        data[ statei ] += 0.5 * numpy.sum( config.MatrixDist * ninj )

    # initialize the sparse matrix with non-zero data stored in the ( row, col ) entries
    H_LongRange = coo_matrix( ( data, ( row, col ) ) , shape=( len( config.Basis ), len( config.Basis ) ) )
    return H_LongRange

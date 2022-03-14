import config
import numpy, sys
from scipy.sparse import coo_matrix

# compute fermionic sign from a hopping process
def ComputeFermionicSign( i, j, conf ) :
    # determine which is larger
    maxsite = max( i, j )
    minsite = min( i, j )

    # return number of particles in conf from minsite+1 up until (but not including) maxsite
    return numpy.power( -1, numpy.sum( conf[ minsite+1 : maxsite ] ) )

# compute the phase factor from the twisted boundary conditions
# # TODO: include twists for tilted square lattice and rectangular slabs on triangular lattice
def ComputeTwist( d, flux ) :
    # 1d chain
    if config.Dimensions == 1 :
        if d == 0 :
            return numpy.exp( -1.0j * ( 2.0 * numpy.pi / config.ns )  *  flux[ 0 ] )
        elif d == 1 :
            return numpy.exp( 1.0j * ( 2.0 * numpy.pi / config.ns )  *  flux[ 0 ] )
        else :
            print( "Bad direction for TBCs in 1d")
            sys.exit()
    elif config.Dimensions == 2 :
        # regular square lattice
        if config.Connectivity == 4 :
            if d == 0 :
                return numpy.exp( 1.0j * ( 2.0 * numpy.pi * flux[ 0 ] / config.lx ) )
            elif d == 1 :
                return numpy.exp( 1.0j * ( 2.0 * numpy.pi * flux[ 1 ] / config.lx ) )
            elif d == 2 :
                return numpy.exp( -1.0j * ( 2.0 * numpy.pi * flux[ 0 ] / config.lx ) )
            elif d == 3 :
                return numpy.exp( -1.0j * ( 2.0 * numpy.pi * flux[ 1 ] / config.lx ) )
            else :
                print("Bad direction for TBCs on square lattice")
                sys.exit()
            return
        # tilted triangular lattice
        elif config.Connectivity == 6 :
            if d == 0 :
                return numpy.exp( 1.0j * ( 2.0 * numpy.pi * ( 2.0 * flux[0] - flux[1] ) / config.lx ) )
            elif d == 1 :
                return numpy.exp( 1.0j * ( 2.0 * numpy.pi * ( flux[0] + flux[1] ) / config.lx ) )
            elif d == 2 :
                return numpy.exp( -1.0j * ( 2.0 * numpy.pi * ( 2.0 * flux[0] - flux[1] ) / config.lx ) )
            elif d == 3 :
                return numpy.exp( -1.0j * ( 2.0 * numpy.pi * ( flux[0] + flux[1] ) / config.lx ) )
            elif d == 4 :
                return numpy.exp( 1.0j * ( 2.0 * numpy.pi * ( -flux[0] + 2.0 * flux[1] ) / config.lx ) )
            elif d == 5 :
                return numpy.exp( -1.0j * ( 2.0 * numpy.pi * ( -flux[0] + 2.0 * flux[1] ) / config.lx ) )
            else :
                print("Bad direction for TBCs on triangular lattice")
                sys.exit()
            return
        else :
            print("Unknown 2d lattice")
            sys.exit()

# create kinetic portion of the Hamiltonian
def CreateKineticHam( Kin , Flux ) :
    row = numpy.array( range( len( config.Basis ) ) )
    col = numpy.array( range( len( config.Basis ) ) )
    data = numpy.zeros( ( len( config.Basis ), ) )

    for s, state in enumerate( config.Basis ) :
        for i in range( len( state ) ) :
            for n in range( config.Connectivity ) :
                if ( state[ i ] == 1 ) and ( state[ config.Neighbors[ i, n ] ] == 0 ) :
                    newstate = numpy.copy( state )
                    newstate[ i ] = 0
                    newstate[ config.Neighbors[ i , n ] ] = 1
                    FerSign = ComputeFermionicSign( i, config.Neighbors[ i, n ] , newstate )
                    Twist = ComputeTwist( n , Flux )
                    newstate_index = numpy.where( numpy.all( config.Basis == newstate , axis=1 ) )

                    data = numpy.append( data, numpy.array( [ -Kin * FerSign * Twist ] ) , axis=0 )
                    row = numpy.append( row, numpy.array( [ s ] ) , axis=0 )
                    col = numpy.append( col, newstate_index[ 0 ] , axis=0 )
                else :
                    pass

    # initialize the sparse matrix with non-zero data stored in the ( row, col ) entries
    H_Kin = coo_matrix( ( data, ( row, col ) ) , shape=( len( config.Basis ), len( config.Basis ) ) )

    return H_Kin

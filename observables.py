import config, kinetic_flux
import numpy, sys
from scipy.sparse import coo_matrix

# compute prefactor for Jx term in Drude weight
def ComputePrefactorx( d ) :
    Prefactorx = 0.0
    u = config.udir
    v = config.vdir

    if config.Connectivity == 2 :
        print("dont have prefactors set up yet for 1d chain")
        sys.exit()
    elif config.Connectivity == 4 :
        print("dont have general prefactors set up yet for square lattice")
        sys.exit()
        if( d == 0 ) :
            Prefactorx = 1.0j * 1.0
        elif( d == 1 ) :
            Prefactorx = 1.0j * 0.0
        elif( d == 2 ) :
            Prefactorx = -1.0j * 1.0
        elif( d == 3 ) :
            Prefactorx = 1.0j * 0.0
    elif config.Connectivity == 6 :
        if( d == 0 ) :
            Prefactorx = ( 1.0j / 1.0 ) * numpy.cos( u * numpy.pi ) + ( 0.0 * 1.0j / 1.0 ) * numpy.sin( u * numpy.pi )
        elif( d == 1 ) :
            Prefactorx = ( 1.0j / 2.0 ) * numpy.cos( u * numpy.pi ) + ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( u * numpy.pi )
        elif( d == 2 ) :
            Prefactorx = ( -1.0j / 1.0 ) * numpy.cos( u * numpy.pi ) - ( 0.0 * 1.0j / 1.0 ) * numpy.sin( u * numpy.pi )
        elif( d == 3 ) :
            Prefactorx = ( -1.0j / 2.0 ) * numpy.cos( u * numpy.pi ) - ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( u * numpy.pi )
        elif( d == 4 ) :
            Prefactorx = ( -1.0j / 2.0 ) * numpy.cos( u * numpy.pi ) + ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( u * numpy.pi )
        elif( d == 5 ) :
            Prefactorx = ( 1.0j / 2.0 ) * numpy.cos( u * numpy.pi ) - ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( u * numpy.pi )
    return Prefactorx

# compute prefactor for Fxx term in Drude weight
def ComputePrefactorxx( d ) :
    Prefactorxx = 0.0
    u = config.udir
    v = config.vdir

    if config.Connectivity == 2 :
        print("dont have prefactors set up yet for 1d chain")
        sys.exit()
    elif config.Connectivity == 4 :
        print("dont have general prefactors set up yet for square lattice")
        sys.exit()
        if( d == 0 ) :
            Prefactorxx = -1.0
        elif( d == 1 ) :
            Prefactorx = 0.0
        elif( d == 2 ) :
            Prefactorx = -1.0
        elif( d == 3 ) :
            Prefactorx = 0.0
    elif config.Connectivity == 6 :
        if( d == 0 ) :
            Prefactorxx = ( ( 1.0j / 1.0 ) * numpy.cos( u * numpy.pi ) + ( 0.0 * 1.0j / 1.0 ) * numpy.sin( u * numpy.pi ) ) * ( ( 1.0j / 1.0 ) * numpy.cos( v * numpy.pi ) + ( 0.0 * 1.0j / 1.0 ) * numpy.sin( v * numpy.pi ) )
        elif( d == 1 ) :
            Prefactorxx = ( ( 1.0j / 2.0 ) * numpy.cos( u * numpy.pi ) + ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( u * numpy.pi ) ) * ( ( 1.0j / 2.0 ) * numpy.cos( v * numpy.pi ) + ( numpy.sqrt( 3.0 ) * 1.0 / 2.0 ) * numpy.sin( v * numpy.pi ) )
        elif( d == 2 ) :
            Prefactorxx = ( ( -1.0j / 1.0 ) * numpy.cos( u * numpy.pi ) - ( 0.0 * 1.0j / 1.0 ) * numpy.sin( u * numpy.pi ) ) * ( ( -1.0j / 1.0 ) * numpy.cos( v * numpy.pi ) - ( 0.0 * 1.0j / 1.0 ) * numpy.sin( v * numpy.pi ) )
        elif( d == 3 ) :
            Prefactorxx = ( ( -1.0j / 2.0 ) * numpy.cos( u * numpy.pi ) - ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( u * numpy.pi ) ) * ( ( -1.0j / 2.0 ) * numpy.cos( v * numpy.pi ) - ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( v * numpy.pi ) )
        elif( d == 4 ) :
            Prefactorxx = ( ( -1.0j / 2.0 ) * numpy.cos( u * numpy.pi ) + ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( u * numpy.pi ) ) * ( ( -1.0j / 2.0 ) * numpy.cos( v * numpy.pi ) + ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( v * numpy.pi ) )
        elif( d == 5 ) :
            Prefactorxx = ( ( 1.0j / 2.0 ) * numpy.cos( u * numpy.pi ) - ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( u * numpy.pi ) ) * ( ( 1.0j / 2.0 ) * numpy.cos( v * numpy.pi ) - ( numpy.sqrt( 3.0 ) * 1.0j / 2.0 ) * numpy.sin( v * numpy.pi ) )
    return Prefactorxx

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

# compute the Drude weight a la Koretsune
def ComputeDrudeWeight( Psi0, Evecs, Evals, Kin, Flux ) :
    Drude = 0.0
    tol = 1.e-3 # tolerance for separating the ground statess

    row = numpy.array( range( len( config.Basis ) ) )
    col = numpy.array( range( len( config.Basis ) ) )
    Jx_data = numpy.zeros( ( len( config.Basis ), ) )
    Fxx_data = numpy.zeros( ( len( config.Basis ), ) )

    for s, state in enumerate( config.Basis ) :
        for i in range( len( state ) ) :
            for n in range( config.Connectivity ) :
                if ( state[ i ] == 1 ) and ( state[ config.Neighbors[ i, n ] ] == 0 ) :
                    newstate = numpy.copy( state )
                    newstate[ i ] = 0
                    newstate[ config.Neighbors[ i , n ] ] = 1
                    FerSign = kinetic_flux.ComputeFermionicSign( i, config.Neighbors[ i, n ] , newstate )
                    Twist = kinetic_flux.ComputeTwist( n , Flux )
                    Prefactorx = ComputePrefactorx( n )
                    Prefactorxx = ComputePrefactorxx( n )
                    newstate_index = numpy.where( numpy.all( config.Basis == newstate , axis=1 ) )

                    Jx_data = numpy.append( Jx_data, numpy.array( [ -Kin * FerSign * Twist * Prefactorx ] ) , axis=0 )
                    Fxx_data = numpy.append( Fxx_data, numpy.array( [ -Kin * FerSign * Twist * Prefactorxx ] ) , axis=0 )
                    row = numpy.append( row, numpy.array( [ s ] ) , axis=0 )
                    col = numpy.append( col, newstate_index[ 0 ] , axis=0 )
                else :
                    pass

    # initialize the sparse matrix with non-zero data stored in the ( row, col ) entries
    JxMatrix = coo_matrix( ( Jx_data, ( row, col ) ) , shape=( len( config.Basis ), len( config.Basis ) ) )
    FxxMatrix = coo_matrix( ( Fxx_data, ( row, col ) ) , shape=( len( config.Basis ), len( config.Basis ) ) )

    # check to see if the matrices are Hermitian
    #Test = FxxMatrix.toarray()
    #for i in range( len( config.Basis ) ) :
    #    for j in range( len( config.Basis ) ) :
    #        if Test[ i, j ] != numpy.conjugate( Test[ j, i ] ) :
    #            print( i, j, Test[i,j], Test[j,i] )

    Fxx = numpy.matmul( numpy.conjugate( Psi0.transpose() ), numpy.matmul( FxxMatrix.toarray() , Psi0 ) ) / ( 2.0 * config.ns )
    Fxx = numpy.real( Fxx )

    Jx = 0.0
    for n in range( 1, Evecs.shape[ 1 ] ) :
        Omega_n = Evals[ n ] - Evals[ 0 ]
        if Omega_n > tol :
            numerator = numpy.linalg.norm( numpy.matmul( Evecs[ :, n ].conjugate().transpose(), numpy.matmul( JxMatrix.toarray(), Psi0 ) ) )**2.0
            Jx += numerator / Omega_n

    Drude = Fxx - ( Jx / config.ns )

    return numpy.array( [ Drude, Fxx, Jx ] )

# compute the optical conductivity
def ComputeOptCond( Psi0, Evecs, Evals, H_Kin, OptCond ) :
    # build current matrix
    CurrentMatrix = 1.0j * H_Kin.toarray()
    tol = 1.e-4

    # loop over the excited states
    for n in range( Evecs.shape[ 1 ] ) :
        Omega_n = Evals[ n ] - Evals[ 0 ]
        if Omega_n > tol :
            ind = numpy.digitize( numpy.array( [ Omega_n ] ), OptCond[ :, 0 ] )
            if ind == len( OptCond[ :, 0 ] ) :
                ind = ind-1
            numerator = numpy.linalg.norm( numpy.matmul( Evecs[ :, n ].conjugate().transpose(), numpy.matmul( CurrentMatrix, Psi0 ) ) )**2.0
            prefactor = numpy.pi / config.ns
            OptCond[ ind, 1 ] += prefactor * numerator / Omega_n

    return OptCond[ :, 1 ]

# compute the density correlation in momentum space
def ComputeDensityCorrelationKSpace( Psi0 ) :
    CorrK = numpy.zeros( ( config.ns , ) )

    for q in range( config.ns ) :
        Corr = 0.0
        for si, state in enumerate( config.Basis ) :
            Op = 0.0
            for i in range( config.ns ) :
                Op += state[ i ] * config.PhiMatrix[ q, i ]
            Corr += numpy.linalg.norm( Op * Psi0[ si ] )**2.0 / config.ns
        CorrK[ q ] += Corr 

    return CorrK

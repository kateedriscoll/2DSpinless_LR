import numpy
from scipy.special import gamma
import mpmath

# create lattice of choice
def CreateLattice( nx, d, z ) :
    global Connectivity ; Connectivity = z
    print("Coordination number: " + str( z ) )

    global Dimensions ; Dimensions = d
    print("Working in " + str( d ) + " dimensions")

    # cluster translation vectors
    global T0 ; T0 = numpy.array( [ 0, 0, 0 ] )
    global T1 ; T1 = numpy.array( [ 0, 0, 0 ] )
    global T2 ; T2 = numpy.array( [ 0, 0, 0 ] )

    # unit vectors
    global u0 ; u0 = numpy.array( [ 0, 0, 0 ] )
    global u1 ; u1 = numpy.array( [ 0, 0, 0 ] )
    global u2 ; u2 = numpy.array( [ 0, 0, 0 ] )

    # lattice vectors real space
    global R0 ; R0 = numpy.array( [ 0, 0, 0 ] )
    global R1 ; R1 = numpy.array( [ 0, 0, 0 ] )
    global R2 ; R2 = numpy.array( [ 0, 0, 0 ] )

    # lattice vectors in k space
    global q0 ; q0 = numpy.array( [ 0, 0, 0 ] )
    global q1 ; q1 = numpy.array( [ 0, 0, 0 ] )
    global q2 ; q2 = numpy.array( [ 0, 0, 0 ] )

    if d == 1 :
        T0[ 0 ] = nx
        u0 = numpy.array( [ 1, 0, 0 ] )

        R0 = T0[0] * u0 + T0[1] * u1 + T0[2] * u2
        R1 = numpy.array( [ 0, 1, 0 ] )
        R2 = numpy.array( [ 0, 0, 1 ] )

        q0 = numpy.array( [ 2. * numpy.pi / nx, 0 ] )
        q1 = numpy.array( [ 0, 0 ] )
        q2 = numpy.array( [ 0, 0 ] )

    else :
        print("not done setting up yet")

    # will never work in d=3 so just save the first 2 parts of the vectors
    u0 = u0[ 0 : 2 ]
    u1 = u1[ 0 : 2 ]
    u2 = u2[ 0 : 2 ]

    # total number of sites
    global ns ; ns = numpy.cross( T0 , T1 )[ 2 ]
    # linear size == 3 * nx
    global lx ; lx = ns // nx

    print( "cluster translation vectors")
    print( T0 )
    print( T1 )
    print( T2 )

    print( "unit vectors")
    print( u0 )
    print( u1 )
    print( u2 )

    print( "lattice vectors (real space)")
    print( R0 )
    print( R1 )
    print( R2 )

    print( "lattice vectors (k space)")
    print( q0 )
    print( q1 )
    print( q2 )

# create table of inverse distances for given length
def CreateEwald( L ) :
    """
    Creates table of inverse distances according to dimensionality and length
    of the system. To be called in the main exact_diagonalization.py with a
    global variable InvDist
    """
    InvDist = numpy.zeros( L , dtype = 'float' )
    d           = 1 # dimension of the system
    Replicas    = 5
    epsilon     = numpy.pi / L
    v_cell =  L
    alpha = 1.0

    const_A = numpy.pi**( d / 2 ) * epsilon**( ( alpha - d ) / 2) / gamma( alpha / 2 ) / v_cell
    const_B = epsilon**( alpha / 2 ) / gamma( alpha  /2 )
    const_C = 2 * epsilon**( alpha / 2 ) / alpha / gamma( alpha / 2 )

    for site in numpy.arange( L ) :
        sum_A = 0.0
        sum_B = 0.0
        r = ( site % L )

        for shx in numpy.arange( - Replicas , Replicas + 1 ) :
            Gl = shx * 2.0 * numpy.pi / L
            nGl = numpy.linalg.norm( Gl )
            if nGl > 1.e-14 :
                sum_A += ( numpy.cos( numpy.dot( Gl , r ) ) - 1 ) * mpmath.expint( - ( ( d - alpha ) / 2 - 1 ) , nGl**2 / 4 / epsilon )

        for shx in numpy.arange( - Replicas , Replicas + 1 ) :
            xl = shx * L
            nxlr = numpy.linalg.norm( xl + r )
            #print(type(alpha),type(nxlr),type(epsilon))
            if nxlr > 1.e-14 :
                sum_B += mpmath.expint( - ( alpha / 2 - 1 ) , nxlr**2 * epsilon )
            nxl = numpy.linalg.norm( xl )
            if nxl > 1.e-14 :
                sum_B -= mpmath.expint( - ( alpha / 2 - 1 ) , nxl**2 * epsilon  )

        InvDist[ site ] = const_A * sum_A + const_B * sum_B + const_C

    return InvDist

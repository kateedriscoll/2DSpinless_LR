import config
import numpy, sys
from scipy.special import gamma
import mpmath
from itertools import product


# create lattice of choice
def CreateLattice( nx, d, z ) :
    config.nx = nx
    config.Connectivity = z
    config.Dimensions = d

    if d == 1 :
        print("1d chain, coordination number: " , z )

        config.T0[ 0 ] = nx
        config.u0 = numpy.array( [ 1, 0, 0 ] )

        config.R0 = config.T0[0] * config.u0 + config.T0[1] * config.u1 + config.T0[2] * config.u2
        config.R1 = numpy.array( [ 0, 1, 0 ] )
        config.R2 = numpy.array( [ 0, 0, 1 ] )

        config.q0 = numpy.array( [ 2. * numpy.pi / config.nx, 0 ] )
        config.q1 = numpy.array( [ 0, 0 ] )
        config.q2 = numpy.array( [ 0, 0 ] )

        # total number of sites
        # using R0 and R1 for 1d chain...should be T0 and T1
        config.ns = numpy.cross( config.R0 , config.R1 )[ 2 ]
        # linear size
        config.lx = config.ns

    elif d==2 and z==4 :
        print("2d square lattice, coordination number: " , z )

        config.T0[ 0 ] = config.nx
        config.T1[ 1 ] = config.nx

        config.u0 = numpy.array( [ 1, 0, 0 ] )
        config.u1 = numpy.array( [ 0, 1 , 0 ] )
        config.u2 = numpy.array( [ 0 , 0, 0 ] )

        config.R0 = config.T0[0] * config.u0 + config.T0[1] * config.u1 + config.T0[2] * config.u2
        config.R1 = config.T1[0] * config.u0 + config.T1[1] * config.u1 + config.T1[2] * config.u2
        config.R2 = numpy.array( [ 0 , 0 , 1 ] )

        config.q0 = ( 2 * numpy.pi * numpy.cross( config.R1 , config.R2 ) / numpy.dot( config.R0 , numpy.cross( config.R1 , config.R2 ) ) )[ 0 : 2 ]
        config.q1 = ( 2 * numpy.pi * numpy.cross( config.R2 , config.R0 ) / numpy.dot( config.R1 , numpy.cross( config.R2 , config.R0 ) ) )[ 0 : 2 ]

        # total number of sites
        config.ns = numpy.cross( config.T0 , config.T1 )[ 2 ]
        # linear size
        config.lx = config.ns // config.nx

    elif d==2 and z==6 :
        print("2d triangular lattice, coordination number: " , z )
        print("tilted cluster")
        config.T0[ 0 ] = config.nx ; config.T0[ 1 ] = config.nx
        config.T1[ 0 ] = -config.nx ; config.T1[ 1 ] = 2*config.nx

        config.u0 = numpy.array( [ 1, 0, 0 ] )
        config.u1 = numpy.array( [ 0.5, 0.5*numpy.sqrt(3.) , 0 ] )
        config.u2 = numpy.array( [ 0 , 0, 0 ] )

        config.R0 = config.T0[0] * config.u0 + config.T0[1] * config.u1 + config.T0[2] * config.u2
        config.R1 = config.T1[0] * config.u0 + config.T1[1] * config.u1 + config.T1[2] * config.u2
        config.R2 = numpy.array( [ 0 , 0 , 1 ] )

        config.q0 = ( 2 * numpy.pi * numpy.cross( config.R1 , config.R2 ) / numpy.dot( config.R0 , numpy.cross( config.R1 , config.R2 ) ) )[ 0 : 2 ]
        config.q1 = ( 2 * numpy.pi * numpy.cross( config.R2 , config.R0 ) / numpy.dot( config.R1 , numpy.cross( config.R2 , config.R0 ) ) )[ 0 : 2 ]

        # total number of sites
        config.ns = numpy.cross( config.T0 , config.T1 )[ 2 ]
        # linear size
        config.lx = config.ns // config.nx
    else :
        print("have not set up this lattice type yet")
        sys.exit()

    # will never work in d=3 so just save the first 2 parts of the vectors
    config.u0 = config.u0[ 0 : 2 ]
    config.u1 = config.u1[ 0 : 2 ]
    config.u2 = config.u2[ 0 : 2 ]

    # neighbors table
    if config.Dimensions == 1 :
        config.Neighbors = numpy.asarray( [ numpy.array( [ ( i -1 ) % config.lx , ( i + 1 ) % config.lx ] ) for i in range( config.ns ) ] )
    elif config.Dimensions == 2 :
        s = numpy.arange( config.ns )
        # neighbor table based on Square lattice
        config.Neighbors = numpy.zeros( ( config.ns, 6 ), dtype = "int" )
        config.Neighbors[ : , 0 ] = ( s + 1 + config.lx ) % config.lx + config.lx * ( s // config.lx )
        config.Neighbors[ : , 1 ] = ( s // config.lx == config.nx - 1 ) * ( ( s - config.nx + config.lx ) % config.lx ) + ( s // config.lx != config.nx - 1 ) * ( s + config.lx )
        config.Neighbors[ : , 2 ] = ( s - 1 + config.lx ) % config.lx + config.lx * ( s // config.lx )
        config.Neighbors[ : , 3 ] = ( s // config.lx == 0 ) * ( config.lx * ( config.nx - 1 ) + ( s + config.nx ) % config.lx ) + ( s // config.lx != 0 ) * ( s - config.lx )
        config.Neighbors[ : , 4 ] = config.Neighbors[ config.Neighbors[ : , 1 ] , 2 ]
        config.Neighbors[ : , 5 ] = config.Neighbors[ config.Neighbors[ : , 0 ] , 3 ]
    else :
        print("Invalid dimensions for neighbor table")
        sys.exit()

# create table of inverse distances for given length
def CreateEwald() :
    """
    Creates table of inverse distances according to dimensionality and length
    of the system. To be called in the main exact_diagonalization.py with a
    global variable InvDist
    """
    config.InvDist = numpy.zeros( config.ns , dtype = 'float' )
    d           = config.Dimensions # dimension of the system
    Replicas    = 5
    epsilon     = 0.
    v_cell      = 0.
    if d == 1 :
        epsilon = numpy.pi / config.ns
        v_cell = config.ns
    elif d == 2 :
        epsilon = ( 3.0 / config.lx )**2.0
        v_cell = numpy.cross( config.R0 , config.R1  )[ 2 ]
    else :
        print("Invalid dimensions")
    alpha = config.Alpha

    const_A = numpy.pi**( d / 2 ) * epsilon**( ( alpha - d ) / 2) / gamma( alpha / 2 ) / v_cell
    const_B = epsilon**( alpha / 2 ) / gamma( alpha  /2 )
    const_C = 2 * epsilon**( alpha / 2 ) / alpha / gamma( alpha / 2 )

    for site in numpy.arange( config.ns ) :
        sum_A = 0.0
        sum_B = 0.0
        r = 0
        if d == 1 :
            r = ( site % config.ns )
            for shx in numpy.arange( - Replicas , Replicas + 1 ) :
                Gl = shx * 2.0 * numpy.pi / config.ns
                nGl = numpy.linalg.norm( Gl )
                if nGl > 1.e-14 :
                    sum_A += ( numpy.cos( numpy.dot( Gl , r ) ) - 1 ) * mpmath.expint( - ( ( d - alpha ) / 2 - 1 ) , nGl**2 / 4 / epsilon )

            for shx in numpy.arange( - Replicas , Replicas + 1 ) :
                xl = shx * config.ns
                nxlr = numpy.linalg.norm( xl + r )
                #print(type(alpha),type(nxlr),type(epsilon))
                if nxlr > 1.e-14 :
                    sum_B += mpmath.expint( - ( alpha / 2 - 1 ) , nxlr**2 * epsilon )
                nxl = numpy.linalg.norm( xl )
                if nxl > 1.e-14 :
                    sum_B -= mpmath.expint( - ( alpha / 2 - 1 ) , nxl**2 * epsilon  )

        elif d == 2 :
            r = ( site % config.lx ) * config.u0 + ( site // config.lx ) * config.u1
            for shx in numpy.arange( - Replicas , Replicas + 1 ) :
                for shy in numpy.arange( - Replicas , Replicas + 1 ) :
                     Gl = shx * config.q0 + shy * config.q1
                     nGl = numpy.linalg.norm( Gl )
                     if nGl > 1.e-14 :
                         sum_A += ( numpy.cos( numpy.dot( Gl , r ) ) - 1 ) * mpmath.expint( - ( ( d - alpha ) / 2 - 1 ) , nGl**2     / 4 / epsilon )

            for shx in numpy.arange( - Replicas , Replicas + 1 ) :
                for shy in numpy.arange( - Replicas , Replicas + 1 ) :
                    xl = shx * config.R0[ 0 : 2 ] + shy * config.R1[ 0 : 2 ]
                    nxlr = numpy.linalg.norm( xl + r )
                    if nxlr > 1.e-14 :
                        sum_B += mpmath.expint( - ( alpha / 2 - 1 ) , nxlr**2 * epsilon )
                    nxl = numpy.linalg.norm( xl )
                    if nxl > 1.e-14 :
                        sum_B -= mpmath.expint( - ( alpha / 2 - 1 ) , nxl**2 * epsilon  )

        else :
            print( "Invalid dimensions in CreateEwald" )
            sys.exit()

        config.InvDist[ site ] = const_A * sum_A + const_B * sum_B + const_C

# create Basis of states with np particles
def CreateBasis( np ) :
    config.np = np
    # generate the full Hilbert space of along the chain
    FullHilbertSpace = numpy.array( list( product( (0,1) , repeat = config.ns )  ) )
    # reduce the size of the Hilbert space to fixed particle sector
    config.Basis = FullHilbertSpace[ numpy.where( numpy.sum( FullHilbertSpace[ : ] , axis=1 ) == config.np ) ]

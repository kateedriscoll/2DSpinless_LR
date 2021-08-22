import numpy
from scipy.special import gamma
import mpmath

# create table of inverse distances for given length
def Ewald( L ) :
    #global InvDist
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

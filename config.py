import numpy

Alpha = 0

Connectivity = 0
Dimensions = 0

# cluster translation vectors
T0 = numpy.array( [ 0, 0, 0 ] )
T1 = numpy.array( [ 0, 0, 0 ] )
T2 = numpy.array( [ 0, 0, 0 ] )

# unit vectors
u0 = numpy.array( [ 0, 0, 0 ] )
u1 = numpy.array( [ 0, 0, 0 ] )
u2 = numpy.array( [ 0, 0, 0 ] )

# lattice vectors (real space)
R0 = numpy.array( [ 0, 0, 0 ] )
R1 = numpy.array( [ 0, 0, 0 ] )
R2 = numpy.array( [ 0, 0, 0 ] )

# lattice vectors (k space)
q0 = numpy.array( [ 0, 0, 0 ] )
q1 = numpy.array( [ 0, 0, 0 ] )
q2 = numpy.array( [ 0, 0, 0 ] )

# number of sites
ns = 0

# linear size of system to set cluster translation vectors
nx = 0

# length of sites along x or y direction
lx = 0
ly = 0

# neighbors table
Neighbors = 0

# inverse distance table and Matrix of inverse distances
InvDist = 0
MatrixDist = 0

# number of paritcles
np = 0

# basis of states
Basis = 0

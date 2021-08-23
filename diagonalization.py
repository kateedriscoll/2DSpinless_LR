import config
import numpy, sys
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

# diagonalize the Hamiltonian matrix
def Diagonalization( Ham , method ) :
    if method == 'Lanczos' :
        evs = min( 50, len(config.Basis)-2 )
        evals, evecs = eigsh( Ham , which='SA', k=evs, tol=1.e-14 )
    elif method == 'Brute' :
        evals, evecs = eigh( Ham.toarray() )
    else :
        print("unknown diagonalization method")
        sys.exit()

    elem = numpy.argsort( evals )
    evals = evals[ elem ]
    evecs = evecs[ :, elem ]

    E0 = evals[ 0 ]
    Psi0 = evecs[ :, 0 ]

    return E0, Psi0, evals, evecs

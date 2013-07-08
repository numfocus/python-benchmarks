# Following http://wiki.cython.org/tutorials/numpy closely.
import numpy as np
cimport numpy as np
cimport cython
#from libc.math cimport sin, cos, atan2, sqrt, pow
cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double tan(double)
    double atan2(double, double)
    double sqrt(double)
    double acos(double)
    double exp(double)
    double abs(double)
    double pow(double, double)

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
def arc_distance(np.ndarray[DTYPE_t, ndim=2] a not None,
                 np.ndarray[DTYPE_t, ndim=2] b not None):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    cdef int a_nrows = a.shape[0]
    cdef int b_nrows = b.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] distance_matrix = np.zeros([a_nrows, b_nrows], dtype = 'float')

    cdef int i, j
    cdef DTYPE_t temp, theta_1, phi_1, theta_2, phi_2
 
    for i in range(a_nrows):
        theta_1 = a[i,0]
        phi_1 = a[i,1]
        for j in range(b_nrows):
            theta_2 = b[j,0]
            phi_2 = b[j,1]
            temp = pow(sin((theta_2-theta_1)/2),2)+cos(theta_1)*cos(theta_2)*pow(sin((phi_2-phi_1)/2),2)
            distance_matrix[i,j] = 2 * (atan2(sqrt(temp),sqrt(1-temp)))
    return distance_matrix

import numpy as np
from math import *

def arc_distance(a, b):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    a_nrows = a.shape[0]
    b_nrows = b.shape[0]

    distance_matrix = np.zeros([a_nrows, b_nrows])
 
    for i in range(a_nrows):
        theta_1 = a[i,0]
        phi_1 = a[i,1]
        for j in range(b_nrows):
            theta_2 = b[j,0]
            phi_2 = b[j,1]
            temp = pow(sin((theta_2-theta_1)/2),2)+cos(theta_1)*cos(theta_2)*pow(sin((phi_2-phi_1)/2),2)
            distance_matrix[i,j] = 2 * (atan2(sqrt(temp),sqrt(1-temp)))
    return distance_matrix
 

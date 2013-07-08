from parakeet import jit
import numpy as np

@jit
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
            temp = np.power(np.sin((theta_2-theta_1)/2),2)+np.cos(theta_1)*np.cos(theta_2)*np.power(np.sin((phi_2-phi_1)/2),2)
            distance_matrix[i,j] = 2 * (np.arctan2(np.sqrt(temp),np.sqrt(1-temp)))
    return distance_matrix
 

import numpy as np

def arc_distance(a, b):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    if (len(a.shape) != 2) or (a.shape[1] != 2):
        raise ValueError('a should be Nx2')
    if (len(b.shape) != 2) or (b.shape[1] != 2):
        raise ValueError('b should be Nx2')
    #  Check for two dimensional arrays
 
    theta_1 = np.tile(a[:,0],(b.shape[0],1)).T
    phi_1 = np.tile(a[:,1],(b.shape[0],1)).T
 
    theta_2 = np.tile(b[:,0],(a.shape[0],1))
    phi_2 = np.tile(b[:,1],(a.shape[0],1))
 
    temp = np.sin((theta_2-theta_1)/2)**2+np.cos(theta_1)*np.cos(theta_2)*np.sin((phi_2-phi_1)/2)**2
    distance_matrix = 2 * (np.arctan2(np.sqrt(temp),np.sqrt(1-temp)))
 
    return distance_matrix    

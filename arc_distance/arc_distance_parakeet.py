# Authors: Alex Rubinsteyn
# License: MIT

from arc_distance import arc_distance_python as adp
from parakeet import jit
import numpy as np 

@jit 
def arc_distance_parakeet_comprehensions(a, b):
  """
  Calculates the pairwise arc distance between all points in vector a and b.
  Uses nested list comprehensions, which are efficiently parallelized
  by Parakeet. 
  """
  def arc_dist(ai, bj):
    theta1 = ai[0]
    phi1 = ai[1]
    theta2 = bj[0]
    phi2 = bj[1]
    d_theta = theta2 - theta1
    d_phi = phi2 - phi1
    temp = (np.sin(d_theta / 2) ** 2) + \
           (np.cos(theta1) * np.cos(theta2) * np.sin(d_phi / 2) ** 2)
    return 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))
  return np.array([[arc_dist(ai, bj) for bj in b] for ai in a])

benchmarks = (("arc_distance_parakeet_for_loops",
               jit(adp.arc_distance_python_nested_for_loops)),
              ("arc_distance_parakeet_comprehensions",
               arc_distance_parakeet_comprehensions)
              )

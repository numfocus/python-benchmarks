# Authors: Olivier Grisel
# License: MIT

from pairwise import pairwise_python
from parakeet import jit
import numpy as np 

def pairwise_parakeet_comprehensions(data):
  return np.array([[np.sqrt(np.sum((a-b)**2)) for b in data] for a in data])

benchmarks = (
    ("pairwise_parakeet_nested_for_loops",
     jit(pairwise_python.pairwise_python_nested_for_loops)),
    ("pairwise_parakeet_inner_numpy",
     jit(pairwise_python.pairwise_python_inner_numpy)),
    ("pairwise_parakeet_comprehensions",
     jit(pairwise_parakeet_comprehensions)),
)

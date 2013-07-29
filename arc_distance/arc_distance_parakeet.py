# Authors: Alex Rubinsteyn
# License: MIT

from arc_distance import arc_distance_python as adp
from parakeet import jit


benchmarks = (("arc_distance_parakeet_for_loops",
               jit(adp.arc_distance_python_nested_for_loops)),
              )

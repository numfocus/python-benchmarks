# Authors: Yuancheng Peng
# License: MIT

from arc_distance import arc_distance_python as adp
from numba import autojit


benchmarks = (("arc_distance_numba_for_loops",
               autojit(adp.arc_distance_python_nested_for_loops)),
              )

# Authors: Olivier Grisel
# License: MIT

from pairwise import pairwise_python
from numba import autojit


benchmarks = (
    ("pairwise_numba_nested_for_loops",
     autojit(pairwise_python.pairwise_python_nested_for_loops)),
)

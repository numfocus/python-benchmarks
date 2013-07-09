# Authors: Olivier Grisel
# License: MIT

from pairwise import pairwise_python
from parakeet import jit


benchmarks = (
    ("pairwise_parakeet_nested_for_loops",
     jit(pairwise_python.pairwise_python_nested_for_loops)),
    ("pairwise_parakeet_inner_broadcasting",
     jit(pairwise_python.pairwise_python_inner_broadcasting)),
)

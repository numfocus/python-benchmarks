from pairwise import pairwise_python
import numba


benchmarks = (
    ("pairwise_numba_nested_for_loops",
     numba.autojit(pairwise_python.pairwise_python_nested_for_loops)),
)

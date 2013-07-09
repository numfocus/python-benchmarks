from julia import julia_python
from numba import autojit


benchmarks = (
    ("julia_numba_for_loops",
     autojit(julia_python.julia_python_for_loops)),
)

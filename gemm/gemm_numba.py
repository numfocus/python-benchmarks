# Authors: Olivier Grisel
# License: MIT

from gemm import gemm_python
from numba import autojit


benchmarks = (
    ("gemm_numba_nested_for_loops",
     autojit(gemm_python.gemm_python_nested_for_loops)),
)

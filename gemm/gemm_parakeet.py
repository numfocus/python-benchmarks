# Authors: Olivier Grisel
# License: MIT

from gemm import gemm_python
from parakeet import jit


benchmarks = (
    ("gemm_parakeet_nested_for_loops",
     jit(gemm_python.gemm_python_nested_for_loops)),
    ("gemm_parakeet_inner_numpy",
     jit(gemm_python.gemm_python_inner_numpy)),
)

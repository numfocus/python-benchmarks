# Author: James Bergstra
# License: MIT

# -- https://github.com/jaberg/python-benchmarks-pyopencl
from pybench_pyopencl import gemm_pyopencl

benchmarks = (
    gemm_pyopencl.gemm_pyopencl_cpu,
)


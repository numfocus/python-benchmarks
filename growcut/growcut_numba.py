from growcut import growcut_python
from numba import autojit


benchmarks = (
    ("growcut_numba",
     autojit(growcut_python.growcut_python)),
)

from growcut import growcut_python
from parakeet import jit


benchmarks = (
    ("growcut_parakeet",
     jit(growcut_python.growcut_python)),
)

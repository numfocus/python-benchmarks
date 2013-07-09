from julia import julia_python
from parakeet import jit

benchmarks = (
    ("julia_parakeet_for_loops",
     jit(julia_python.julia_python_for_loops)),
    ("julia_parakeet_numpy",
     jit(julia_python.julia_python_numpy)),
)

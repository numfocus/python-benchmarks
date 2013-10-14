from julia import julia_python
from parakeet import jit

benchmarks = (
    ("julia_parakeet_for_loops",
     jit(julia_python.julia_python_for_loops)),

    # Can't run the NumPy version under Parakeet since the following
    # features are not supported: 
    #   - np.seterr 
    #   - np.ogrid
    #   - complex numbers 
    #("julia_parakeet_numpy",
    # jit(julia_python.julia_python_numpy)),
)

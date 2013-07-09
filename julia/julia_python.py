# Authors: Kurt W. Smith, Serge Guelton
# License: MIT

import numpy as np

def kernel(zr, zi, cr, ci, lim, cutoff):
    ''' Computes the number of iterations `n` such that 
        |z_n| > `lim`, where `z_n = z_{n-1}**2 + c`.
    '''
    count = 0
    while ((zr*zr + zi*zi) < (lim*lim)) and count < cutoff:
        zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
        count += 1
    return count

def julia_python_for_loops(cr, ci, N, bound=1.5, lim=1000., cutoff=1e6):
    ''' Pure Python calculation of the Julia set for a given `c`.  No NumPy
        array operations are used.
    '''
    julia = np.empty((N, N), dtype=np.uint32)
    grid_x = np.linspace(-bound, bound, N)
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_x):
            julia[i,j] = kernel(x, y, cr, ci, lim, cutoff=cutoff)
    return julia

def julia_python_numpy(cr, ci, N, bound=1.5, lim=4., cutoff=1e6):
    ''' Pure Python calculation of the Julia set for a given `c` using NumPy
    array operations.
    '''
    c = cr + 1j * ci
    orig_err = np.seterr()
    np.seterr(over='ignore', invalid='ignore')
    julia = np.zeros((N, N), dtype=np.uint32)
    X, Y = np.ogrid[-bound:bound:N*1j, -bound:bound:N*1j]
    iterations = X + Y * 1j
    count = 1
    while not np.all(julia) and count < cutoff:
        mask = np.logical_not(julia) & (np.abs(iterations) >= lim)
        julia[mask] = count
        count += 1
        iterations = iterations**2 + c
    if count == cutoff:
        julia[np.logical_not(julia)] = count
    np.seterr(**orig_err)
    return julia

benchmarks = (
    julia_python_for_loops,
    julia_python_numpy,
)

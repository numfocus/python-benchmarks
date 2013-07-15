# Authors: James Bergstra
# License: MIT

import numpy as np
import time
import pyopencl as cl
import numpy

mf = cl.mem_flags

PROFILING = 0

ctx = cl.create_some_context()
if PROFILING:
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)
else:
    queue = cl.CommandQueue(ctx)

_cache = {}

def pairwise_pyopencl_cpu_prepare(shp, dtype):
    N, D = shp
    ctype = {
            'float32': 'float',
            'float64': 'double',
            }[str(dtype)]

    prg = cl.Program(ctx, """
        __kernel void foo(__global %(ctype)s *a, __global %(ctype)s *c)
        {
          for(int n0 = get_global_id(0); n0 < %(N)s; n0 += get_global_size(0))
          {
              for(int m0 = get_global_id(1); m0 < %(N)s; m0 += get_global_size(1))
              {
                  __global %(ctype)s *an = a + n0 * %(D)s;
                  __global %(ctype)s *am = a + m0 * %(D)s;
                  %(ctype)s buf = 0;
                  for (int d = 0; d < %(D)s; ++d)
                  {
                    %(ctype)s diff = am[d] - an[d];
                    buf += diff * diff;
                  }
                  c[m0 * %(N)s + n0] = sqrt(buf);
              }
          }
        }
        """ % locals()).build()

    return prg.foo


comptimes = []
def pairwise_pyopencl_cpu(data):
    data = np.asarray(data, order='C')
    N, D = data.shape
    rval = np.empty((N, N), dtype=data.dtype)
    try:
        f = _cache[(data.shape, data.dtype)]
    except:
        f = pairwise_pyopencl_cpu_prepare(data.shape, data.dtype)
        _cache[(data.shape, data.dtype)] = f
    data_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=data)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, rval.nbytes)
    ev = f(queue, (N, 1), (4, 1), data_buf, dest_buf)
    cl.enqueue_copy(queue, rval, dest_buf)
    queue.finish()
    if PROFILING:
        comptimes.append(1e-9 * (ev.profile.end - ev.profile.start))
        print 'computation time', min(comptimes)
    return rval


benchmarks = (
    pairwise_pyopencl_cpu,
)

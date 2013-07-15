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

    odd_d = "" if 0 == D % 2 else """
    __global %(ctype)s * a1 = (__global %(ctype)s*) (a);
    %(ctype)s diff = a1[(n0 + 1) * %(D)s - 1] - a1[(m0 + 1) * %(D)s - 1];
    buf.s0 += diff * diff;
    """

    prg = cl.Program(ctx, """
        __kernel void lower(__global %(ctype)s2 *a, __global %(ctype)s *c)
        {
          for(int n0 = get_global_id(0); n0 < %(N)s; n0 += get_global_size(0))
          {
              for(int m0 = get_global_id(1); m0 < %(N)s; m0 += get_global_size(1))
              {
                if (n0 < m0) continue;
                  __global %(ctype)s2 *an = a + n0 * %(D)s / 2;
                  __global %(ctype)s2 *am = a + m0 * %(D)s / 2;
                  %(ctype)s2 buf = 0;
                  for (int d = 0; d < %(D)s/2; ++d)
                  {
                    %(ctype)s2 diff = am[d] - an[d];
                    buf += diff * diff;
                  }
                  %(odd_d)s;
                  c[m0 * %(N)s + n0] = sqrt(buf.s0 + buf.s1);
              }
          }
        }
        __kernel void upper(__global %(ctype)s *a, __global %(ctype)s *c)
        {
          for(int n0 = get_global_id(0); n0 < %(N)s; n0 += get_global_size(0))
          {
              for(int m0 = get_global_id(1); m0 < %(N)s; m0 += get_global_size(1))
              {
                if (n0 >= m0) continue;
                  c[m0 * %(N)s + n0] = c[n0 * %(N)s + m0];
              }
          }
        }
        """ % locals()).build()

    return prg.lower, prg.upper


comptimes = []
def pairwise_pyopencl_cpu(data):
    data = np.asarray(data, order='C')
    N, D = data.shape
    try:
        lower, upper = _cache[(data.shape, data.dtype)]
    except:
        lower, upper = pairwise_pyopencl_cpu_prepare(data.shape, data.dtype)
        _cache[(data.shape, data.dtype)] = lower, upper
    data_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=data)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, N * N * data.dtype.itemsize)
    try:
        rval, _ = cl.enqueue_map_buffer(queue, dest_buf, cl.map_flags.READ,
                offset=0, shape=(N, N), dtype=data.dtype)
        need_copy = False
    except TypeError: #OSX's OCL needs this?
        rval = np.empty((N, N), dtype=data.dtype)
        need_copy = True
    lower(queue, (N, 1), (1, 1), data_buf, dest_buf)
    upper(queue, (4, 4), (1, 1), data_buf, dest_buf)
    if need_copy:
        cl.enqueue_copy(queue, rval, dest_buf)
    else:
        queue.finish()
    if PROFILING:
        comptimes.append(1e-9 * (ev.profile.end - ev.profile.start))
        print 'computation time', min(comptimes)
    return rval


benchmarks = (
    pairwise_pyopencl_cpu,
)

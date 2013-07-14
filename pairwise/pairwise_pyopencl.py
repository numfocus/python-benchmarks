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

def pairwise_pyopencl_cpu_prepare(shp, dtype, BM, BN):
    N, D = shp
    ctype = {
            'float32': 'float',
            'float64': 'double',
            }[str(dtype)]

    options = "-cl-mad-enable -cl-fast-relaxed-math -cl-unsafe-math-optimizations"

    prg = cl.Program(ctx, """
        __kernel void foo(__global %(ctype)s *a, __global %(ctype)s *c)
        {
          int n0 = get_global_id(0) * %(BN)s;
          int m0 = get_global_id(1) * %(BM)s;
          __global %(ctype)s *an = a + n0 * %(D)s;
          __global %(ctype)s *am = a + m0 * %(D)s;
          %(ctype)s buf[%(BN)s * %(BM)s];
          for (int ii = 0; ii < %(BM)s * %(BN)s; ++ii) buf[ii] = 0;
          for (int d = 0; d < %(D)s; ++d)
          {
              for (int bm = 0; bm < %(BM)s; ++bm)
              {
                  for (int bn = 0; bn < %(BN)s; ++bn)
                  {
                    %(ctype)s diff = am[bm * %(D)s + d] - an[bn * %(D)s + d];
                    buf[bm * %(BN)s + bn] += diff * diff;
                  }
              }
          }
          for (int bm = 0; bm < %(BM)s; ++bm)
          {
              for (int bn = 0; bn < %(BN)s; ++bn)
              {
                  c[(m0 + bm) * %(N)s + n0 + bn] = sqrt(buf[bm * %(BM)s + bn]);
              }
          }
        }
        """ % locals()).build(options)

    return prg.foo


def pairwise_pyopencl_cpu(data):
    data = np.asarray(data, order='C')
    N, D = data.shape
    if N % 2 == 0:
        m_block_size = 2
        n_block_size = 4 if (0 == N % 4) else 2
    else:
        m_block_size = n_block_size = 1
    m_blocks = int(np.ceil(N / float(m_block_size)))
    n_blocks = int(np.ceil(N / float(n_block_size)))
    rval = np.empty((N, N), dtype=data.dtype)
    try:
        f = _cache[(data.shape, data.dtype)]
    except:
        f = pairwise_pyopencl_cpu_prepare(data.shape, data.dtype,
                m_block_size, n_block_size)
        _cache[(data.shape, data.dtype)] = f
    data_buf = cl.Buffer(ctx, mf.USE_HOST_PTR, hostbuf=data)
    dest_buf = cl.Buffer(ctx, mf.USE_HOST_PTR, hostbuf=rval)
    ev = f(queue, (n_blocks, m_blocks), None, data_buf, dest_buf)
    ev.wait()
    if PROFILING:
        print 'computation time', 1e-9 * (ev.profile.end - ev.profile.start)
    return rval


if PROFILING:

    _data = numpy.random.rand(200, 100).astype(numpy.float32)

    times = []
    for i in range(50):
        t0 = time.time()
        pairwise_pyopencl_cpu(_data)
        t1 = time.time()
        times.append(t1 - t0)
    print 'best time', min(times)


benchmarks = (
    pairwise_pyopencl_cpu,
)

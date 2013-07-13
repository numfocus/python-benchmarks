import numpy as np
import time
import pyopencl as cl
import numpy

mf = cl.mem_flags

BLOCK_SIZE = 32

PROFILING = 0

ctx = cl.create_some_context()
if PROFILING:
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)
else:
    queue = cl.CommandQueue(ctx)

_cache = {}

def pairwise_ocl_cpu_prepare(shp, dtype):
    N, D = shp
    ctype = {
            'float32': 'float',
            'float64': 'double',
            }[str(dtype)]
    B = BLOCK_SIZE

    prg = cl.Program(ctx, """
        __kernel void foo(__global %(ctype)s *a, __global %(ctype)s *c)
        {
          int m0 = get_global_id(0) * %(B)s;
          int n0 = get_global_id(1) * %(B)s;
          for (int m = m0; m < min(m0 + %(B)s, %(N)s); ++m)
          {
              for (int n = n0; n < min(n0 + %(B)s, %(N)s); ++n)
              {
                  %(ctype)s diff = 0;
                  %(ctype)s sum = 0;
                  for (int d = 0; d < %(D)s; ++d)
                  {
                    diff = a[n * %(D)s + d] - a[m * %(D)s + d];
                    sum += diff * diff;
                  }
                  c[n * %(N)s + m] = sqrt(sum);
              }
          }
        }
        """ % locals()).build()

    return prg.foo


def pairwise_ocl_cpu(data):
    data = np.asarray(data, order='C')
    N, D = data.shape
    blocks = int(np.ceil(N / float(BLOCK_SIZE)))
    rval = np.empty((N, N), dtype=data.dtype)
    try:
        f = _cache[(data.shape, data.dtype)]
    except:
        f = pairwise_ocl_cpu_prepare(data.shape, data.dtype)
        _cache[(data.shape, data.dtype)] = f
    data_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=data)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, rval.nbytes)
    ev = f(queue, (blocks, blocks), None, data_buf, dest_buf)
    ev.wait()
    if PROFILING:
        print 'computation time', 1e-9 * (ev.profile.end - ev.profile.start)


if 0:

    _data = numpy.random.rand(200, 100).astype(numpy.float32)

    t0 = time.time()
    pairwise_ocl_cpu(_data)
    t1 = time.time()
    print 'runtime', (t1 - t0)

    t0 = time.time()
    pairwise_ocl_cpu(_data)
    t1 = time.time()
    print 'runtime', (t1 - t0)

benchmarks = (
    pairwise_ocl_cpu,
)

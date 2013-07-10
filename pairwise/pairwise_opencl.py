import numpy as np
import time
import pyopencl as cl
import numpy
import numpy.linalg as la

mf = cl.mem_flags

BLOCK_SIZE = 16

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

_cache = {}

def pairwise_ocl_cpu_prepare(N16, shp, dtype):
    N, D = shp
    ctype = {
            'float32': 'float',
            'float64': 'double',
            }[str(dtype)]
    B = BLOCK_SIZE

    prg = cl.Program(ctx, """
        __kernel void foo(__global %(ctype)s *a)
        {
          int m0 = get_global_id(0) * %(B)s;
          int n0 = get_global_id(1) * %(B)s;
          __global %(ctype)s *c = a + %(D)s * %(N16)s;
          for (int m = m0; m < min(m0 + %(B)s, %(N)s); ++m)
          {
              for (int n = n0; n < min(n0 + %(B)s, %(N)s); ++n)
              {
                  %(ctype)s diff = 0;
                  %(ctype)s sum = 0;
                  for (int d = 0; d < %(D)s; ++d)
                  {
                    diff = a[n * %(D)s + d] - a[m * %(D)s + d];
                    //diff = a[n + d * %(N)s] - a[m + d * %(N)s];
                    sum += diff * diff;
                  }
                  c[n * %(N)s + m] = sqrt(sum);
              }
          }
        }
        """ % locals()).build()

    return prg.foo

def pairwise_ocl_cpu(data):
    N, D = data.shape
    blocks = int(np.ceil(N / float(BLOCK_SIZE)))
    N16 = blocks * 16
    data16 = np.empty((N16, D + N16), order='F', dtype=data.dtype)
    data16[:N, :D] = data
    try:
        f = _cache[(N16, data.shape, data.dtype)]
    except:
        f = pairwise_ocl_cpu_prepare(N16, data.shape, data.dtype)
        _cache[(N16, data.shape, data.dtype)] = f
    a_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=data16)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, data16.nbytes)
    f(queue, (blocks, blocks), None, a_buf)
    queue.flush()



_data = numpy.random.rand(1000, 3).astype(numpy.float32)

t0 = time.time()
pairwise_ocl_cpu(_data)
t1 = time.time()
print 'runtime', (t1 - t0)

t0 = time.time()
pairwise_ocl_cpu(_data)
t1 = time.time()
print 'runtime', (t1 - t0)


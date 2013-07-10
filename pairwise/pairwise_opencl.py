import numpy as np
import time
import pyopencl as cl
import numpy
import numpy.linalg as la
N = 1000
D = 3

a = numpy.random.rand(N, D).astype(numpy.float32)
c = numpy.random.rand(N, N).astype(numpy.float32)



def pairwise_ocl_cpu_prepare(dtype):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    def pairwise_ocl_cpu(data):



mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
#b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

prg = cl.Program(ctx, """
    __kernel void sum(__global const float *a,
    __global float *c)
    {
      int m0 = get_global_id(0) * %(N)s;
      int n0 = get_global_id(1) * %(N)s;
      for (int m = 0; m < %(N)s; ++m)
      {
      }
      float diff = 0;
      float sum = 0;
      for (int d = 0; d < %(D)s; ++d)
      {
        diff = a[n * %(D)s + d] - a[m * %(D)s + d];
        //diff = a[n + d * %(N)s] - a[m + d * %(N)s];
        sum += diff * diff;
      }
      c[n * %(N)s + m] = sqrt(sum);
    }
    """ % locals()).build()

f = prg.sum
f.set_args(a_buf, dest_buf)
t0 = time.time()
for i in range(100):
    ev = cl.enqueue_nd_range_kernel(queue, f, (N, N), None)
ev.wait()
t1 = time.time()
print 'cl took', (t1 - t0) * 1000 / 100, 'ms per iter'

t0 = time.time()
for i in range(10):
    np.sqrt(((a[:, None, :] - a) ** 2).sum(-1))
t1 = time.time()
print 'np took', (t1 - t0) * 1000 / 10, 'ms per iter'



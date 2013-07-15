# Authors: James Bergstra
# License: MIT

import numpy as np
import pyopencl as cl

mf = cl.mem_flags

PROFILING = 0

_cache = {}

def julia_cpu_prepare(cr, ci, N, bound, lim, cutoff):
    ctx = cl.create_some_context()
    if PROFILING:
        queue = cl.CommandQueue(
            ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(ctx)

    grid_x = np.linspace(-bound, bound, N)
    x0 = grid_x[0]
    dx = grid_x[1] - grid_x[0]
    limlim = lim ** 2

    prg = cl.Program(ctx, """
        __kernel void foo(__global long *outbuf)
        {
          int ii = get_global_id(0);
          int jj = get_global_id(1);

          double zr = %(x0)s + ii * %(dx)s;
          double zi = %(x0)s + jj * %(dx)s;
          long count = 0;

          while (((zr*zr + zi*zi) < %(limlim)s) && (count < %(cutoff)s))
          {
              double tmp = zr * zr - zi * zi + %(cr)s;
              zi = 2. * zr * zi + %(ci)s;
              zr = tmp;
              count += 1;
          }
          outbuf[ii * %(N)s + jj] = count;
          }

        """ % locals()).build()

    return prg.foo, queue, ctx


def julia_pyopencl(cr, ci, N, bound=1.5, lim=4., cutoff=1e6):
    output = np.empty((N, N), dtype='int64')
    args = (cr, ci, N, bound, lim, cutoff)
    try:
        f, queue, ctx = _cache[args]
    except:
        f, queue, ctx = julia_cpu_prepare(*args)
        _cache[args] = f, queue, ctx
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)
    ev = f(queue, (N, N), None, dest_buf)
    if PROFILING:
        ev.wait()
        print 'computation time', 1e-9 * (ev.profile.end - ev.profile.start)
    cl.enqueue_copy(queue, output, dest_buf)
    return output


benchmarks = (
    julia_pyopencl,
)

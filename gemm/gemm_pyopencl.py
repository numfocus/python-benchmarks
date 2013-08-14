# Authors: James Bergstra
# License: BSD-3

import sys
import time

from mako.template import Template
import numpy as np
import pyopencl as cl

mf = cl.mem_flags

PROFILING = 0

ctx = cl.create_some_context()
if PROFILING:
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)
else:
    queue = cl.CommandQueue(ctx)


def ctype_from_dtype(dtype):
    return {
            'float32': 'float',
            'float64': 'double',
            }[str(dtype)]


def elemstrides(strides, dtype, vec=1, vecdim=-1):
    size = {
        'float32': 4,
        'float64': 8,
    }[str(dtype)]
    # TODO: raise StrideError
    for stride in strides:
        assert stride % size == 0
    val_strides = tuple(int(s / size) for s in strides)
    if vec == 1:
        return val_strides
    vecdim = range(len(strides))[vecdim]   # -- make vecdim non-neg
    for ii, val_stride in enumerate(val_strides):
        if ii == vecdim:
            assert val_stride == 1
        else:
            assert val_stride % vec == 0
    vec_strides = [int(s / vec) for s in val_strides]
    vec_strides[vecdim] = 1
    return tuple(vec_strides)



def memoize(f):
    cache = {}
    def new_fn(*args, **kwargs):
        key = args + tuple(sorted(kwargs.items()))
        try:
            return cache[key]
        except KeyError:
            rval = f(*args, **kwargs)
            cache[key] = rval
            return rval
    new_fn.__name__ = f.__name__
    new_fn.memoize_cache = cache
    return new_fn


@memoize
def gemm_cpu_prepare_reference(alpha, beta, M, N, K, dtype,
                               Astrides, Bstrides, Cstrides):
    ctype = ctype_from_dtype(dtype)
    (As0, As1) = elemstrides(Astrides, dtype)
    (Bs0, Bs1) = elemstrides(Bstrides, dtype)
    (Cs0, Cs1) = elemstrides(Cstrides, dtype)
    prg = cl.Program(ctx, """
        __kernel void ref(__global %(ctype)s *A,
                          __global %(ctype)s *B,
                          __global %(ctype)s *C)
        {
          for(int mm = get_global_id(0); mm < %(M)s; mm += get_global_size(0))
          {
            for(int nn = get_global_id(1); nn < %(N)s; nn += get_global_size(1))
            {
              %(ctype)s buf = 0;
              for (int kk = 0; kk < %(K)s; ++kk)
              {
                  buf += A[mm * %(As0)s + kk * %(As1)s]
                       * B[kk * %(Bs0)s + nn * %(Bs1)s];
              }
              C[mm * %(Cs0)s + nn * %(Cs1)s] *= %(beta)s;
              C[mm * %(Cs0)s + nn * %(Cs1)s] += %(alpha)s * buf;
            }
          }
        }
        """ % locals()).build()

    return prg.ref

class StrideError(Exception):
    """StrideError"""

class BlockingError(Exception):
    """BlockingError"""

vectorized_text =  """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void kern(__global ${ctype}${KB} *A,
                  __global ${ctype}${NB} *B,
                  __global ${ctype}${NB} *C)
{
  const ${ctype}${NB} beta = (${ctype}${NB})(
      ${beta}
  % for ii in range(NB - 1):
      , ${beta}
  % endfor
  );
  ${ctype}${NB} tmp;
  % for ii in range(MB):
  ${ctype}${KB} Abuf${ii};
  ${ctype}${NB} Cbuf${ii};
  % endfor
  ${ctype}${NB} Bbuf;
  for(int mb = get_global_id(0); mb < ${NoMB}; mb += get_global_size(0))
  {
    for(int nb = get_global_id(1); nb < ${NoNB}; nb += get_global_size(1))
    {
      % for ii in range(MB):
      Cbuf${ii} = (${ctype}${NB})(
                    0
                    % for foo in range(NB - 1):
                    , 0
                    % endfor
                    );
      % endfor

      for (int kb = 0; kb < ${NoKB}; ++kb)
      {
        // load KB columns of A at a time
        % for ii in range(MB):
        Abuf${ii} = A[${As0} * (mb * ${MB} + ${ii}) + kb];
        % endfor

        % for kki in range(KB):
        Bbuf = B[(kb * ${KB} + ${kki}) * ${Bs0} + nb];

            % for ii in range(MB):

            tmp = (${ctype}${NB})(
                    Abuf${ii}.s${kki}
                    % for foo in range(NB - 1):
                    , Abuf${ii}.s${kki}
                    % endfor
                    );
            Cbuf${ii} = mad(tmp, Bbuf, Cbuf${ii});
            % endfor

        % endfor
      }

      % if alpha != 1:
          % for ii in range(MB):
              Cbuf${ii} *= (${ctype}${NB})(
                    ${alpha}
                    % for foo in range(NB - 1):
                    , ${alpha}
                    % endfor
                    );
          % endfor
      % endif

      % for ii in range(MB):
          % if beta == 0:
              C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] = Cbuf${ii};
          % elif beta == 1:
              C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] += Cbuf${ii};
          % else:
              C[(mb * ${MB} + ${ii}) * ${Cs0} + nb] = mad(beta, C[(mb* ${MB} + ${ii}) * ${Cs0} + nb], Cbuf${ii});
          % endif
      % endfor
    }
  }
}
    """


@memoize
def gemm_cpu_prepare_vectorized(alpha, beta, M, N, K, dtype,
                               Astrides, Bstrides, Cstrides, MB, NB, KB):
    ctype = ctype_from_dtype(dtype)
    (As0, As1) = elemstrides(Astrides, dtype, MB)
    (Bs0, Bs1) = elemstrides(Bstrides, dtype, NB)
    (Cs0, Cs1) = elemstrides(Cstrides, dtype, MB)
    if MB: assert As1 == 1
    if NB: assert Bs1 == 1
    if MB: assert Cs1 == 1
    NoMB = M // MB
    NoNB = N // NB
    NoKB = K // KB
    if M != MB * NoMB:
        raise BlockingError()
    if N != NB * NoNB:
        raise BlockingError()
    if K != KB * NoKB:
        raise BlockingError()
    text = Template(vectorized_text, output_encoding='ascii').render(**locals())
    for ii, line in enumerate(text.split('\n')):
        print ii, line
    prg = cl.Program(ctx, text).build()
    print 'built!'

    return prg.kern


comptimes = []
def gemm_pyopencl_cpu(alpha, A, B, beta, C):
    kern = None
    global_shape = (4, 4)   # enough for different cores
    local_shape = (1, 1)    # I think this does nothing on CPU (true?)

    # TODO: some values of these constants that should work, do not.
    # e.g. 16, 8, 8 -> wrong answer
    #      4, 4, 8  -> segfault
    for MB in [8]:
        for NB in [8]:
            for KB in [8]:
                if kern:
                    continue
                try:
                    kern = gemm_cpu_prepare_vectorized(
                        alpha, beta,
                        C.shape[0], C.shape[1], A.shape[1],
                        A.dtype,
                        A.strides, B.strides, C.strides,
                        MB=MB, NB=NB, KB=KB)
                    print 'Using kernel for MB=%i NB=%i KB=%i' % (MB, NB, KB)
                except StrideError:
                    pass
    if kern is None:
        kern = gemm_cpu_prepare_reference(alpha, beta,
                                          C.shape[0], C.shape[1], A.shape[1],
                                          A.dtype,
                                          A.strides, B.strides, C.strides)

    A_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=C)
    ev = kern(queue, global_shape, local_shape, A_buf, B_buf, C_buf)
    cl.enqueue_copy(queue, C, C_buf)
    queue.finish()
    if PROFILING:
        comptimes.append(1e-9 * (ev.profile.end - ev.profile.start))
        print 'computation time', min(comptimes)


benchmarks = (
    gemm_pyopencl_cpu,
)

NNN = 1024 * 2
def main(shape=(NNN, NNN, NNN), seed=0, dtype=np.float32):
    rng = np.random.RandomState(seed)
    A = np.asarray(rng.normal(size=(shape[0], shape[1])), dtype=dtype)
    B = np.asarray(rng.normal(size=(shape[1], shape[2])), dtype=dtype)
    C = np.asarray(rng.normal(size=(shape[0], shape[2])), dtype=dtype)
    C2 = np.empty_like(C)
    Z = np.ones(10 * 1024 * 1024)
    alpha = 1.0
    beta = 0.0
    FLOPS = shape[0] * shape[1] * (shape[2] + 1) * 2
    for i in range(500):
        rng.seed(i)
        C[:] = np.asarray(rng.normal(size=(shape[0], shape[2])), dtype=dtype)

        t0 = time.time()
        gemm_pyopencl_cpu(alpha, A, B, beta, C)
        t1 = time.time()
        print 'pyopencl time: ', (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)
        continue

        rng.seed(i)
        C2[:] = np.asarray(rng.normal(size=(shape[0], shape[2])), dtype=dtype)

        t0 = time.time()
        C3 = alpha * np.dot( A, B) + beta * C2
        t1 = time.time()
        print 'np.dot time:   ', (t1 - t0), 'GFlops', FLOPS / (t1 - t0) / (1000 ** 3)

        print np.max(abs(C - C3))
        assert np.allclose(C, C3)
        # -- clear processor cache
        Z.sum()
        time.sleep(1)

if __name__ == '__main__':
    sys.exit(main())

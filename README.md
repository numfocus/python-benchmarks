python-benchmarks
=================

A set of benchmark problems and implementations for Python.

This repository is the result of a discussion of started by @aterrel at SciPy
2013 where people interested in the development of compiler technologies for
the Python programming language shared design decisions.

The goal of this repository to gather Python implementations of realistic use
cases where:

- naive code written with the [CPython](http://python.org) interpreter is too
  slow to be of practical use,

- an implementation of the algorithm cannot be efficiently vectorized using
  NumPy primitives (for instance by involving nested `for`-loops)

Initial use case focus on **data processing** tasks such as **machine
learning** and **signal processing**.

For each benchmark, we would like to gather:

- a naive pure python implementation (optionally using NumPy for large
  homogeneous numerical datastructures) run using CPython

- variants of the Python version that should be able to run the naive pure
  Python version with minimal code change:

  - JIT compiler packaged as a library for CPython such as:
      - [Numba](http://numba.pydata.org/) or
      - [Parakeet](http://iskandr.github.io/parakeet/)

  - JIT compiler implemented in an alternative Python interpreter such as:
      - [PyPy](http://pypy.org/) optionally with
        [NumPyPy](https://bitbucket.org/pypy/numpypy)

  - Python to C/C++ code translation + compiled extension for the CPython
    interpreter such as done by:
      - [Pythran](https://github.com/serge-sans-paille/pythran)
      - [Shed Skin](http://code.google.com/p/shedskin/)

- pure Python programs that explicitly represent the computation as a graph of
  Python objects and use code generation and a compiler to dynamically build a
  compiled extension such as done by [Theano](https://github.com/Theano/Theano)

- alternative language implementations in Cython, C or Fortran with Python
  bindings to serve as speed reference.

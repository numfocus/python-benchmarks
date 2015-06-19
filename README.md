python-benchmarks
=================

A set of benchmark problems and implementations for Python.


Results
-------

[numfocus.github.io/python-benchmarks](
  http://numfocus.github.io/python-benchmarks)


Motivation
----------

This repository is the result of a discussion started by
[@aterrel](https://github.com/aterrel) at SciPy 2013 where people interested in
the development of compiler technologies for the Python programming language
shared design decisions.

The goal of this repository to gather Python implementations of realistic use
cases where:

- naive code written with the [CPython](http://python.org) interpreter is too
  slow to be of practical use,

- an implementation of the algorithm cannot be efficiently vectorized using
  NumPy primitives (for instance by involving nested `for`-loops)

Initial use cases focus on **data processing** tasks such as **machine
learning** and **signal processing**.

For each benchmark, we would like to gather:

- a naive pure python implementation (optionally using NumPy for large
  homogeneous numerical datastructures) run using CPython

- variants of the Python version that should be able to run the naive pure
  Python version with minimal code change:

  - JIT compiler packaged as a library for CPython such as:
      - [Numba](http://numba.pydata.org/) or
      - [Parakeet](http://www.parakeetpython.com)

  - JIT compiler implemented in an alternative Python interpreter such as:
      - [PyPy](http://pypy.org/) optionally with
        [NumPyPy](https://bitbucket.org/pypy/numpypy)

  - Python to C/C++ code translation + compiled extension for the CPython
    interpreter such as done by:
      - [Cython](http://cython.org/)
      - [Pythran](https://github.com/serge-sans-paille/pythran)
      - [Shed Skin](http://code.google.com/p/shedskin/)

- pure Python programs that explicitly represent the computation as a graph of
  Python objects and use code generation and a compiler to dynamically build a
  compiled extension such as done by [Theano](https://github.com/Theano/Theano)

- alternative language implementations in Cython, C or Fortran with Python
  bindings to serve as speed reference.


Running
-------

To run all the benchmarks:

    python run_benchmarks.py

To run the benchmarks of a specific folder:

    python run_benchmarks.py --folders pairwise

To run only the benchmarks with specific platforms:

    python run_benchmarks.py --platforms numba parakeet cython

To ignore previously collected data:

    python run_benchmarks.py --ignore-data

To see all the tracebacks of the collected errors:

    python run_benchmarks.py --log-level debug

To open a browser on the generated HTML report page:

    python run_benchmarks.py --open-report

To publish the generated report to github (assuming you want to push to
origin):

    make github

Or to another remote alias:

    WEB_ALIAS_REPO=upstream make github


Dependencies
------------

### Using pip

- Some dependencies use [llvmpy](http://www.llvmpy.org/) that require to have
  llvm built with the `REQUIRES_RTTI=1` environment variable. Under OSX you
  can install llvm with HomeBrew:

    brew install llvm --rtti

- Install the dependencies from the `requirements.txt` file:

    pip install -r requirements.txt

Note: some packages (pythran and ply) have a depency on SciPy which is
complicated and slow to install from source because of the need of a gfortran
compiler and a large C++ code base. It is recommended to install a binary
package for SciPy (see http://scipy.org/install.html for instructions).


### Using conda / Anaconda

TODO

### Non CPython dependencies

You can also install PyPy from http://pypy.org

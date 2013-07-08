# Authors: Olivier Grisel
# License: MIT
from __future__ import print_function

from collections import OrderedDict
import os
from time import time
import logging

from jinja2 import Template

log = logging.getLogger(__name__)


MAIN_REPORT_TEMPLATE = """
Benchmark Results
=================

{% for result in bench_results %}

{{ result.group_name }}
-----------------------

TODO: add link to the sources code on github

===================== ============= ============= ===============
Function name         Cold time (s) Warm time (s) Error
===================== ============= ============= ===============

{% for record in records %}



{% endfor %}


Runtime Environment
===================

Hardware
--------

TODO: describe host machine (CPU, RAM, ...) with psutil

Software
--------

TODO: list version numbers for all the libraries and compilers.

"""


def find_benchmarks(folders=None, platforms=None):
    """Collect benchmarks collable and shared environment initializers."""

    benchmark_groups = []

    if folders is None:
        here = os.path.dirname(os.path.abspath(__file__))
        dir_content = os.listdir(here)
        folders = [f for f in dir_content
                   if (os.path.isdir(f) and
                       os.path.exists(os.path.join(f, '__init__.py')))]
        folders.sort()

    for folder in folders:
        group_name = os.path.basename(folder)
        collected_benchmarks = []
        modules_in_error = []

        pkg = __import__(group_name, fromlist="dummy")

        benchmark_groups.append((
            group_name,
            getattr(pkg, 'make_env', None),
            collected_benchmarks,
            modules_in_error,
        ))

        for module_filename in sorted(os.listdir(folder)):
            module_name, ext = os.path.splitext(module_filename)
            if ext not in ('.py', '.so', '.dll'):
                continue
            module_name = "%s.%s" % (group_name, module_name)
            module = __import__(module_name, fromlist="dummy")

            for benchmark in getattr(module, 'benchmarks', ()):
                if callable(benchmark):
                    collected_benchmarks.append(
                        (benchmark.__name__, benchmark))
                elif isinstance(benchmark, tuple) and len(benchmark) == 2:
                    collected_benchmarks.append(benchmark)
                else:
                    raise ValueError("Found invalid benchmark %r in %s" %
                                     benchmark, module_name)
    return benchmark_groups


def run_benchmark(func, args, kwargs, memory=False):
    """Call a function with the provided arguments"""
    # TODO: find a way to use memory_profiler on non-python, builtin functions
    tic = time()
    func(*args, **kwargs)
    toc = time()
    return toc - tic


def run_benchmarks(folders=None, platforms=None, catch_errors=True,
                   memory=True):
    collected = find_benchmarks(folders=folders, platforms=platforms)

    bench_results = []

    for group_name, make_env, benchmarks, import_errors in collected:
        logging.info("Running benchmark group %s", group_name)
        args, kwargs = make_env() if make_env is not None else {}
        print(group_name)
        print('-' * len(group_name) + '\n')
        print("Benchmarks:")

        records = []
        for name, func in benchmarks:
            logging.info("Benchmarking %s", name)
            try:
                cold_time = run_benchmark(func, args, kwargs, memory)
                warm_time = run_benchmark(func, args, kwargs, memory)
                runtime_error = None
            except Exception as e:
                cold_time, warm_time = None, None
                if catch_errors:
                    runtime_error = e
                else:
                    raise
            records.append(OrderedDict(
                name=name,
                source_url=None,  # TODO
                cold_time=cold_time,
                warm_time=warm_time,
                runtime_error=runtime_error))
            # TODO: add special support for PyPy with a sub-process

        bench_results.append(OrderedDict(
            group_name=group_name,
            source_url=None,  # TODO
            records=records,
            import_errors=import_errors,
        ))

    return bench_results


if __name__ == "__main__":
    # TODO: use argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    bench_results = run_benchmarks(catch_errors=True, memory=True)
    print(Template(MAIN_REPORT_TEMPLATE).render(
        bench_results=bench_results,
        runtime_environment=None,
    ))

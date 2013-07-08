# Authors: Olivier Grisel
# License: MIT
from __future__ import print_function
import os
from time import time


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
    for group_name, make_env, benchmarks, errors in collected:
        args, kwargs = make_env() if make_env is not None else {}
        print(group_name)
        print('-' * len(group_name) + '\n')
        print("Benchmarks:")
        for name, func in benchmarks:
            try:
                cold_time = run_benchmark(func, args, kwargs, memory)
                warm_time = run_benchmark(func, args, kwargs, memory)
                print("- %s:\t cold: %0.3fs\t| warm: %0.3fs" %
                      (name, cold_time, warm_time))
            except Exception as e:
                if catch_errors:
                    print("- %s: %r" % (name, e))
                else:
                    raise
        print()
        if len(errors) > 0:
            print("Import errors:")
            for name, error in errors:
                print("- %s: %s" % (name, error))
            print()


if __name__ == "__main__":
    # TODO: use argparse
    run_benchmarks(catch_errors=True, memory=True)

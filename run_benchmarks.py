# Authors: Olivier Grisel
# License: MIT
from __future__ import print_function
import os


def find_benchmarks(folders=None, platforms=None):
    """Collect benchmarks collable and shared environment initializers."""

    benchmark_groups = []

    if folders is None:
        here = os.path.dirname(os.path.abspath(__file__))
        dir_content = os.listdir(here)
        folders = [f for f in dir_content
                   if (os.path.isdir(f) and
                       os.path.exists(os.path.join(f, '__init__.py')))]

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

        for module_filename in os.listdir(folder):
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


if __name__ == "__main__":
    # TODO: use argparse
    for group_name, make_env, benchmarks, errors in find_benchmarks():
        print(group_name)
        print('-' * len(group_name) + '\n')
        print("Benchmarks:")
        for name, func in benchmarks:
            print("- " + name)
        print()
        if len(errors) > 0:
            print("Import errors:")
            for name, error in errors:
                print("- %s: %s" % (name, error))
            print()

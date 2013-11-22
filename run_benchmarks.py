# Authors: Olivier Grisel
# License: MIT
from __future__ import print_function

import argparse
try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict
import json
import os
import traceback
from time import time
import logging

from jinja2 import Template
import numpy as np
import matplotlib.pyplot as plt

# imports for machine stats 
import multiprocessing 
import platform
from types import FunctionType 

try:
    # Use automated Cython support when available
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
except ImportError:
    pass

log = logging.getLogger("run_benchmarks")

REPORT_FILENAME = 'report/index.html'
DATA_FILENAME = 'report/benchmark_results.json'

GROUP_URL_PATTERN = ("https://github.com/numfocus/python-benchmarks/"
                     "tree/master/%s")

MODULE_URL_PATTERN = ("https://github.com/numfocus/python-benchmarks/"
                      "tree/master/%s")

ABOUT_URL = ("https://github.com/numfocus/python-benchmarks/blob/master/"
             "README.md#motivation")

GITHUB_REPO_URL = "https://github.com/numfocus/python-benchmarks"


MAIN_REPORT_TEMPLATE_FILENAME = "templates/index.html"

LOG_FORMAT = '%(asctime)s %(levelname)-8s %(name)-8s %(message)s'


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

        benchmark_groups.append(OrderedDict([
            ('name', group_name),
            ('make_env', getattr(pkg, 'make_env', None)),
            ('benchmarks', collected_benchmarks),
            ('import_errors', modules_in_error),
        ]))

        for module_filename in sorted(os.listdir(folder)):
            module_name, ext = os.path.splitext(module_filename)
            if ext and ext not in ('.py', '.so', '.dll', '.pyx'):
                continue

            if not module_name.startswith(group_name + "_"):
                continue

            platform_name = module_name[len(group_name) + 1:]
            if platforms is not None and platform_name not in platforms:
                continue

            abs_module_name = "%s.%s" % (group_name, module_name)

            try:
                module = __import__(abs_module_name, fromlist="dummy")
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                log.error("Failed to load %s: %s: %s", abs_module_name,
                          type(e).__name__, e)
                tb = traceback.format_exc()
                loading_error = OrderedDict([
                    ('name', module_name),
                    ('error_type', error_type),
                    ('error_message', error_message),
                    ('traceback', tb),
                ])
                modules_in_error.append(loading_error)
                module = None

            module_source = "%s/%s" % (group_name, module_filename)
            for benchmark in getattr(module, 'benchmarks', ()):
                if callable(benchmark):
                    collected_benchmarks.append(
                        (module_source, benchmark.__name__, benchmark))
                elif isinstance(benchmark, tuple) and len(benchmark) == 2:
                    collected_benchmarks.append((module_source,) + benchmark)
                else:
                    raise ValueError("Found invalid benchmark %r in %s" %
                                     benchmark, module_name)
    return benchmark_groups



def run_benchmark(name, func, args, kwargs, memory=False, n_runs=5,
                  slow_threshold=1):
    """Call a function with the provided arguments"""
    # TODO: find a way to use memory_profiler on non-python, builtin functions
    def time_once():
        tic = time()
        func(*args, **kwargs)
        toc = time()
        return toc - tic

    first_timing = time_once()
    # if we're running a user-defined pure Python function, assume there's no warmup
    if isinstance(func, FunctionType):
        cold = None 
        warm = first_timing
        all_warm_timings = [first_timing]
    else:
        # Give a warm/cold time for every benchmark, even if there's no JIT
        # Take the best time of several runs for fast executions
      other_timings = []
      for i in range(n_runs - 1):
        t = time_once()
        other_timings.append(t)

      all_warm_timings = other_timings
      best_warm_timing = np.min(all_warm_timings)
      cold = first_timing
      warm = best_warm_timing

    return OrderedDict([
        ('name', name),
        ('cold_time', cold),
        ('warm_time', warm),
        ('all_warm_times', all_warm_timings),
        ('std_warm_times', np.std(all_warm_timings)),
    ])


def run_benchmarks(folders=None, platforms=None, catch_errors=True,
                   memory=True):
    collected = find_benchmarks(folders=folders, platforms=platforms)

    bench_results = []

    for group in collected:
        log.info("Running benchmark group %s", group['name'])
        make_env = group.get('make_env')
        args, kwargs = make_env() if make_env is not None else ((), {})

        records = []
        runtime_errors = []
        for module_source, name, func in group['benchmarks']:
            log.info("Benchmarking %s", name)
            module_source_url = MODULE_URL_PATTERN % module_source
            try:
                record = run_benchmark(name, func, args, kwargs, memory=memory)
                record['source_url'] = module_source_url
                records.append(record)
                log.info("%s: cold: %s, warm: %s",
                         name, record['cold_time'], record['warm_time'])
            except Exception as e:
                if catch_errors:
                    error_type = type(e).__name__
                    error_message = str(e)
                    tb = traceback.format_exc()
                    runtime_error = OrderedDict([
                        ('name', name),
                        ('source_url', module_source_url),
                        ('error_type', error_type),
                        ('error_message', error_message),
                        ('traceback', tb),
                    ])
                    runtime_errors.append(runtime_error)
                    log.warn("Could not run %s: %s: %s", name,
                             error_type, e)
                    log.debug(tb)
                else:
                    raise

            # TODO: add special support for PyPy with a sub-process

        # Rerank records by ascending warm time:
        if len(records) > 0:
            records.sort(key=lambda r: r['warm_time'])
            slowest_time = records[-1]['warm_time']
            for rank, record in enumerate(records):
                record['rank'] = rank + 1  # start at 1 instead of 0
                record['speedup'] = slowest_time / record['warm_time']

        bench_results.append(OrderedDict([
            ('group_name', group['name']),
            ('source_url', GROUP_URL_PATTERN % group['name']),
            ('records', records),
            ('runtime_errors', runtime_errors),
            ('import_errors', group['import_errors']),
        ]))
    return bench_results


def plot_group(group, width=0.5, zoom_scale=None, log_scale=False,
               figsize=(12, 6), folder="report/images"):
    records = group['records']
    if len(records) == 0:
        return

    name = group['group_name']
    for r in records:
        r['max_time'] = r['cold_time'] or r['warm_time']

    labels = [r['name'][len(name) + 1:] for r in records]
    best_time = records[0]['warm_time']
    warm_time = np.asarray([r['warm_time'] for r in records])
    std = np.asarray([r['std_warm_times'] for r in records])
    max_time = np.asarray([r['cold_time'] or r['warm_time'] for r in records])

    if log_scale:
        bottom = best_time / 2.
    else:
        bottom = 0.0

    plt.figure(figsize=figsize)
    ind = np.arange(len(labels))
    p1 = plt.bar(ind, max_time, width, color='g', alpha=0.2, log=log_scale,
                 bottom=bottom)
    p2 = plt.bar(ind, warm_time, width, yerr=[np.zeros(len(std)), std],
                 color='g', ecolor='g', alpha=0.4, log=log_scale,
                 bottom=bottom)

    title = group['group_name']
    if log_scale:
        title += " (log scale)"
    elif zoom_scale:
        title += " (zoom %dx best time)" % zoom_scale
    plt.title(title)
    plt.ylabel('Time (s)')
    plt.xticks(ind + width / 2., labels, rotation=10)

    if log_scale:
        plt.yscale('log')
        plt.ylim(bottom, None)
    else:
        if zoom_scale:
            plt.ylim((0, warm_time[0] * zoom_scale))
        else:
            plt.ylim(0, max(max_time) * 1.1)

    plt.legend((p2[0], p1[0]),
               ('Execution time', 'Cold startup overhead'),
               loc='upper left')

    # Save the image in the report folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    if log_scale:
        suffix = "_logscale"
    elif zoom_scale:
        suffix = "_zoom_%dx_best" % zoom_scale if zoom_scale else ""
    else:
        suffix = ""

    filename = "%s%s.png" % (group['group_name'], suffix)
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath)

    # Make it available to the template engine
    group.setdefault('plot_filenames', []).append(filename)


def build_report(bench_data, report_filename=REPORT_FILENAME,
                 data_filename=DATA_FILENAME):
    for group in bench_data['benchmark_results']:
        plot_group(group, zoom_scale=5, log_scale=False)
        plot_group(group, zoom_scale=None, log_scale=True)
    with open(MAIN_REPORT_TEMPLATE_FILENAME, 'rb') as f:
        rendered = Template(f.read()).render(
            bench_results=bench_data['benchmark_results'],
            bech_env=bench_data['benchmark_environment'],
            about_url=ABOUT_URL,
            github_repo_url=GITHUB_REPO_URL,
            json_data_url=os.path.basename(bench_data_filename),
            sysinfo = {
              'platform': platform.platform(),
              'python_version': platform.python_version(), 
              'processor' : platform.processor(), 
              'cpu_count' : multiprocessing.cpu_count()
            }
        )
    report_filename = 'report/index.html'
    log.info("Writing report to: %s", report_filename)
    with open(report_filename, 'wb') as f:
        f.write(rendered)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-catch-errors', action='store_true',
                        default=False)
    parser.add_argument('--folders', nargs='*', default=None)
    parser.add_argument('--platforms', nargs='*', default=None)
    parser.add_argument('--ignore-data', action='store_true', default=False)
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--open-report', action='store_true',
                        default=False)
    return parser.parse_args(args)

if __name__ == "__main__":
    import sys
    options = parse_args(sys.argv[1:])

    log_level = getattr(logging, options.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format=LOG_FORMAT)

    bench_data_filename = DATA_FILENAME
    if os.path.exists(bench_data_filename) and not options.ignore_data:
        log.info("Loading bench data from: %s", bench_data_filename)
        with open(bench_data_filename, 'rb') as f:
            bench_data = json.load(f)
    else:
        bench_results = run_benchmarks(
            catch_errors=not options.no_catch_errors,
            folders=options.folders,
            platforms=options.platforms,
        )
        bench_environment = {}  # TODO
        bench_data = OrderedDict([
            ('benchmark_results', bench_results),
            ('benchmark_environment', bench_environment),
        ])
        log.info("Writing bench data to: %s", bench_data_filename)
        with open(bench_data_filename, 'wb') as f:
            json.dump(bench_data, f, indent=2)
    report_filename = os.path.abspath('report/index.html')
    build_report(bench_data, data_filename=bench_data_filename,
                 report_filename=report_filename)
    if options.open_report:
        import webbrowser
        webbrowser.open("file://" + report_filename)

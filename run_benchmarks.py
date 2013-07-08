# Authors: Olivier Grisel
# License: MIT
from __future__ import print_function

from collections import OrderedDict
import os
from time import time
import logging

from jinja2 import Template

log = logging.getLogger(__name__)

GROUP_URL_PATTERN = ("https://github.com/numfocus/python-benchmarks/"
                     "tree/master/%s")

ABOUT_URL = ("https://github.com/numfocus/python-benchmarks/blob/master/"
             "README.md#motivation")

GITHUB_REPO_URL = "https://github.com/numfocus/python-benchmarks"


MAIN_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="css/bootstrap.min.css" rel="stylesheet" media="screen">
</head>
<body>
<div class="container">

<ul class="nav nav-pills">
  <li><a href="{{ github_repo_url }}">Github repo</a></li>
  <li><a href="{{ about_url }}">About</a></li>
</ul>

<h2>Python Benchmark Results</h2>

{% for result in bench_results %}

<h3><a href="{{ result.source_url }}" title="View source">
    {{ result.group_name }}</a></h3>

<table class="table">
<thead>
<tr>
    <th>Function name</th>
    <th>Cold time (s)</th>
    <th>Warm time (s)</th>
    <th>Runtime Error</th>
</tr>
</thead>
<tbody>
{% for record in result.records %}
<tr>
    <td><i class="icon-tasks"></i> {{ record.name }}</td>
    <td>{% if record.cold_time %}
        {{ "{:0.3f}".format(record.cold_time) }}
        {% else %}
        N/A
        {% endif %}
    </td>
    <td>{% if record.warm_time %}
        {{ "{:0.3f}".format(record.warm_time) }}
        {% else %}
        N/A
        {% endif %}
    </td>
    <td>{% if record.runtime_error %}
        {{ record.runtime_error }}
        {% else %}
        N/A
        {% endif %}
    </td>
</tr>
{% endfor %}
</tbody>
</table>

<p>TODO: log import error summary here</p>

{% endfor %}

<h2>Runtime Environment</h2>

<h3>Hardware</h3>

TODO: describe host machine (CPU, RAM, ...) with psutil

<h3>Software</h3>


TODO: list version numbers for all the libraries and compilers.

</div>
<script src="js/bootstrap.min.js"></script>
</body>
</html>
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
        log.info("Running benchmark group %s", group_name)
        args, kwargs = make_env() if make_env is not None else {}

        records = []
        for name, func in benchmarks:
            log.info("Benchmarking %s", name)
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

            record = OrderedDict([
                ('name', name),
                ('cold_time', cold_time),
                ('warm_time', warm_time),
                ('runtime_error', runtime_error),
            ])
            records.append(record)
            log.info(", ".join("%s: %r" % (k, v) for k, v in record.items()))

            # TODO: add special support for PyPy with a sub-process

        bench_results.append(OrderedDict([
            ('group_name', group_name),
            ('source_url', GROUP_URL_PATTERN % group_name),
            ('records', records),
            ('import_errors', import_errors),
        ]))
    return bench_results


if __name__ == "__main__":
    # TODO: use argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    bench_results = run_benchmarks(catch_errors=True, memory=True)
    rendered = Template(MAIN_REPORT_TEMPLATE).render(
        bench_results=bench_results,
        runtime_environment=None,
        about_url=ABOUT_URL,
        github_repo_url=GITHUB_REPO_URL,
    )
    report_filename = 'report/index.html'
    log.info("Writing report to: %s", report_filename)
    with open(report_filename, 'wb') as f:
        f.write(rendered)

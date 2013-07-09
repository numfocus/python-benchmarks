# Authors: Olivier Grisel
# License: MIT
from __future__ import print_function

from collections import OrderedDict
import json
import os
import traceback
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
  <link href="css/custom.css" rel="stylesheet" media="screen">
</head>
<body>
<div class="container">

<ul class="nav nav-pills">
  <li><a href="{{ github_repo_url }}">Github repo</a></li>
  <li><a href="{{ json_data_url }}">JSON data</a></li>
  <li><a href="{{ about_url }}">About</a></li>
</ul>

<h2>Python Benchmark Results</h2>

{% for result in bench_results %}

<section id="results-{{ result.group_name }}">
<h3>{{ result.group_name }}
    (<a href="{{ result.source_url }}"
     title="Source code for {{ result.group_name }}">source code</a>)
    <a title="Permalink to this headline"
       href="#results-{{ result.group_name }}"
       class="headerlink">&para;</a>
</h3>

<table class="table table-striped table-hover">
<colgroup>
  <col class="bench_icon">
  <col class="bench_name">
  <col class="bench_time">
  <col class="bench_time">
</colgroup>
<thead>
<tr>
  <th></th>
  <th>Function name</th>
  <th>Cold time (s)</th>
  <th>Warm time (s)</th>
</tr>
</thead>
<tbody>
{% for record in result.records %}
<tr>
  <td><i class="icon-tasks"></i></td>
  <td>{{ record.name }}</td>
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
</tr>
{% endfor %}
</tbody>
</table>
</section>
{% endfor %}

<h2>Runtime Environment</h2>

<h3>Hardware</h3>

TODO: describe host machine (CPU, RAM, ...) with psutil

<h3>Software</h3>

TODO: list version numbers for all the libraries and compilers.

<h2>Errors</h2>

{% for result in bench_results %}
{%if result.import_errors or result.runtime_errors %}

<section id="errors-{{ result.group_name }}">
<h3>{{ result.group_name }}
    <a title="Permalink to this headline"
       href="#errors-{{ result.group_name }}"
       class="headerlink">&para;</a></h3>

{%if result.import_errors  %}
<div id="import-errors-{{ result.group_name }}">
<h4>Benchmark loading errors
    <a title="Permalink to this headline"
       href="#import-errors-{{ result.group_name }}"
       class="headerlink">&para;</a></h4>
<dl>
  {% for import_error in result.import_errors %}
    <dt>{{ import_error.name }}: {{ import_error.error_type }}</dt>
    <dd>
      <pre class="pre-scrollable">{{ import_error.traceback }}</pre>
    </dd>
  {% endfor %}
</dl>
</div>
{% endif %}

{%if result.runtime_errors  %}
<div id="runtime-errors-{{ result.group_name }}">
<h4>Benchmark execution errors
    <a title="Permalink to this headline"
       href="#runtime-errors-{{ result.group_name }}"
       class="headerlink">&para;</a></h4>
<dl>
  {% for runtime_error in result.runtime_errors %}
    <dt>{{ runtime_error.name }}: {{ runtime_error.error_type }}</dt>
    <dd>
      <pre class="pre-scrollable">{{ runtime_error.traceback }}</pre>
    </dd>
  {% endfor %}
</dl>
</div>
{% endif %}

</section>
{% endif %}
{% endfor %}

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
            if ext not in ('.py', '.so', '.dll', '.pyx'):
                continue
            module_name = "%s.%s" % (group_name, module_name)

            try:
                module = __import__(module_name, fromlist="dummy")
            except Exception as e:
                log.error("Failed to load %s: %s", module_name, e)
                tb = traceback.format_exc()
                loading_error = OrderedDict([
                    ('name', module_name),
                    ('error_type', type(e).__name__),
                    ('error', str(e)),
                    ('traceback', tb),
                ])
                modules_in_error.append(loading_error)

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
        runtime_errors = []
        for name, func in benchmarks:
            log.info("Benchmarking %s", name)
            try:
                cold_time = run_benchmark(func, args, kwargs, memory)
                warm_time = run_benchmark(func, args, kwargs, memory)
                record = OrderedDict([
                    ('name', name),
                    ('cold_time', cold_time),
                    ('warm_time', warm_time),
                ])
                records.append(record)
                log.info("%s: cold: %0.3fs, warm: %0.3fs",
                         name, cold_time, warm_time)
            except Exception as e:
                if catch_errors:
                    tb = traceback.format_exc()
                    runtime_error = OrderedDict([
                        ('name', name),
                        ('error_type', type(e).__name__),
                        ('error', str(e)),
                        ('traceback', tb),
                    ])
                    runtime_errors.append(runtime_error)
                    log.warn("Could not run %s: %s", name, e)
                    log.debug(tb)
                else:
                    raise

            # TODO: add special support for PyPy with a sub-process

        bench_results.append(OrderedDict([
            ('group_name', group_name),
            ('source_url', GROUP_URL_PATTERN % group_name),
            ('records', records),
            ('runtime_errors', runtime_errors),
            ('import_errors', import_errors),
        ]))
    return bench_results


if __name__ == "__main__":
    # TODO: use argparse
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')
    bench_data_filename = 'report/benchmark_results.json'
    if os.path.exists(bench_data_filename):
        log.info("Loading bench data from: %s", bench_data_filename)
        with open(bench_data_filename, 'rb') as f:
            bench_results = json.load(f)
    else:
        bench_results = run_benchmarks(catch_errors=True, memory=True)
        log.info("Writing bench data to: %s", bench_data_filename)
        with open(bench_data_filename, 'wb') as f:
            json.dump(bench_results, f, indent=2)

    rendered = Template(MAIN_REPORT_TEMPLATE).render(
        bench_results=bench_results,
        runtime_environment=None,
        about_url=ABOUT_URL,
        github_repo_url=GITHUB_REPO_URL,
        json_data_url=os.path.basename(bench_data_filename),
    )
    report_filename = 'report/index.html'
    log.info("Writing report to: %s", report_filename)
    with open(report_filename, 'wb') as f:
        f.write(rendered)

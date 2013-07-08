import sys
import os
import timeit

setup = lambda module: 'from {} import julia'.format(module)
stmt = 'julia(0.285, 0.01, 200, 1.5)'

for module_path in sys.argv[1:]:
    module, _ = os.path.splitext(module_path)
    timer = timeit.Timer(stmt, setup(module))
    print "{}\t{}".format(module, min(timer.repeat(number=3)))

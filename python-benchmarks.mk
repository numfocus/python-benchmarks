# programs
PYTHON=python
PYTHON_CONFIG=python-config
PYTHRAN=pythran
CYTHON=cython

# compiler flags
CFLAGS=-O2 -fopenmp
CXXFLAGS=$(CFLAGS)

# modules
MODULES=$(sort $(patsubst %, $(TARGET)_%, $(BACKENDS)))

# targets
all:$(MODULES) $(TARGET).py
	$(PYTHON) $(TARGET).py $(MODULES) | tee $(TARGET).time

clean:
	rm -rf $(MODULES) *.pyc $(TARGET).time

# auto rules
%.so:%.py
	$(PYTHRAN) $< $(CXXFLAGS) -o $@

%.pyo:%.py
	$(PYTHON) -O -c "import py_compile ; py_compile.compile('$<')"

%.c:%.pyx
	$(CYTHON) $<

%.so:%.c
	$(CC) `$(PYTHON_CONFIG) --cflags --ldflags` -fPIC -shared $(CFLAGS) $< -o $@

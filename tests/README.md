# Code Coverage
## Coverage source
The "source" for code coverage defines a list of files over which code coverage is checked. This is defined in the `HERON/check_py_coverage.sh` script. By default when the source directory is specified, coverage.py measures coverage over all files in the source directory(ies) ending with .py, .pyw, .pyo, or .pyc that have typical punctuation. It also measures all files in subdirectories that also include an `__init__.py` file. For details see https://coverage.readthedocs.io/en/7.5.4/source.html#source

HERON code coverage is currently set up to run all files in the `HERON/src/` directory as well as in the `HERON/templates/` directory (provided the limitations listed above). Exceptions, which are in these directories but not covered, are listed as omitted files and directories in `HERON/check_py_coverage.sh`. Currently this list is comprised of the following files:
- `HERON/src/ARMABypass.py`
- `HERON/src/dispatch/twin_pyomo_test.py`
- `HERON/src/dispatch/twin_pyomo_test_rte.py`
- `HERON/src/dispatch/twin_pyomo_limited_ramp.py`

Note additionally that files in some subdirectories of `HERON/src` are omitted automatically by coverage.py because those subdirectories lack an `__init__.py` file. An example is the `HERON/src/Testers/` directory.
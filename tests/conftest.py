# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  pytest hooks for HERON tests. Adds the --no-raven option, which runs the tests marked raven_free in the
  files listed under raven_free_files in pytest.ini, with RAVEN, TEAL, and FARM imports blocked.
"""
import os
import sys

import pytest

# the HERON tests import the package as "HERON", so its parent directory must be importable
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_HERON_PARENT = os.path.abspath(os.path.join(_TESTS_DIR, '..', '..'))
if _HERON_PARENT not in sys.path:
  sys.path.insert(0, _HERON_PARENT)
if _TESTS_DIR not in sys.path:
  sys.path.insert(0, _TESTS_DIR)
import raven_blocker # pylint: disable=wrong-import-position

def pytest_addoption(parser):
  """
    Registers the --no-raven command line option and the raven_free_files ini setting
    @ In, parser, pytest.Parser, pytest's option parser
    @ Out, None
  """
  parser.addoption('--no-raven', action='store_true', default=False,
                   help='block RAVEN imports and run only raven_free tests in the files listed in raven_free_files')
  parser.addini('raven_free_files', type='paths',
                help='test files whose imports work without RAVEN; only these are collected with --no-raven')

def pytest_configure(config):
  """
    With --no-raven, blocks RAVEN imports before any test module is collected and selects raven_free tests
    @ In, config, pytest.Config, pytest configuration
    @ Out, None
  """
  if not config.getoption('no_raven'):
    return
  raven_blocker.block_raven()
  config.option.markexpr = 'raven_free'
  config._heron_raven_free_files = {os.path.abspath(str(p)) for p in config.getini('raven_free_files')}
  missing = sorted(p for p in config._heron_raven_free_files if not os.path.isfile(p))
  if missing:
    raise pytest.UsageError(f'raven_free_files lists files that do not exist: {missing}')

def pytest_ignore_collect(collection_path, config):
  """
    With --no-raven, skips collecting test files not listed in raven_free_files
    @ In, collection_path, pathlib.Path, path pytest is about to collect
    @ In, config, pytest.Config, pytest configuration
    @ Out, ignore, bool or None, True to skip the path; None to defer to pytest
  """
  if not config.getoption('no_raven') or collection_path.is_dir():
    return None
  if collection_path.suffix != '.py' or os.path.abspath(str(collection_path)) in config._heron_raven_free_files:
    return None
  return True

# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Checks that HERON can be imported and reports clearly when RAVEN is not available. Each check runs in a
  fresh Python process with RAVEN imports blocked, so these tests are meaningful even where RAVEN is installed.
"""
import os
import sys
import json
import subprocess
import textwrap
import unittest

import pytest

pytestmark = pytest.mark.raven_free

TESTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
HERON_PARENT = os.path.abspath(os.path.join(TESTS_DIR, os.pardir, os.pardir))

def run_without_raven(code):
  """
    Runs code in a new Python process with RAVEN, TEAL, and FARM imports blocked
    @ In, code, str, Python source to run; it should print a single JSON value as its last line of output
    @ Out, result, object, the decoded JSON value printed by the code
  """
  script = 'import raven_blocker\nraven_blocker.block_raven()\n' + textwrap.dedent(code)
  env = dict(os.environ)
  env['PYTHONPATH'] = os.pathsep.join([TESTS_DIR, HERON_PARENT, env.get('PYTHONPATH', '')])
  proc = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    raise AssertionError(f'subprocess failed (exit {proc.returncode}):\n{proc.stderr}')
  return json.loads(proc.stdout.strip().splitlines()[-1])


class TestImportWithoutRaven(unittest.TestCase):
  """ HERON packages that do not need RAVEN can be imported without it """

  def test_import_heron(self):
    """
      The HERON package and HERON.src import, without loading the RAVEN plugin entities
      @ In, None
      @ Out, None
    """
    result = run_without_raven("""
      import json
      import HERON
      import HERON.src
      print(json.dumps({'plugins': hasattr(HERON, 'DispatchManager') or hasattr(HERON, 'DispatchPlot')}))
    """)
    self.assertFalse(result['plugins'])

  def test_import_snippets(self):
    """
      The template snippets import; only the RAVEN-generated distribution classes are absent
      @ In, None
      @ Out, None
    """
    result = run_without_raven("""
      import json
      from HERON.templates.snippets import distributions, RavenSnippet
      from HERON.templates.snippets.factory import factory
      print(json.dumps({'uniform': hasattr(distributions, 'Uniform'),
                        'registered': len(factory.registered_classes)}))
    """)
    self.assertFalse(result['uniform'])
    self.assertGreater(result['registered'], 0)

  def test_templates_require_raven(self):
    """
      Importing the RAVEN workflow templates fails with RavenNotFoundError
      @ In, None
      @ Out, None
    """
    result = run_without_raven("""
      import json
      from HERON.src import _utils as hutils
      try:
        import HERON.templates.imports
        print(json.dumps('imported'))
      except hutils.RavenNotFoundError:
        print(json.dumps('RavenNotFoundError'))
    """)
    self.assertEqual(result, 'RavenNotFoundError')


class TestRavenHelpers(unittest.TestCase):
  """ The _utils helpers for locating RAVEN and TEAL behave correctly when they are missing """

  def test_optional_lookups(self):
    """
      Optional lookups return False and leave sys.path as it was
      @ In, None
      @ Out, None
    """
    result = run_without_raven("""
      import sys, json
      from HERON.src import _utils as hutils
      before = list(sys.path)
      found = [hutils.ensure_raven_on_path(required=False), hutils.ensure_teal_on_path(required=False),
               hutils.raven_available()]
      print(json.dumps({'found': found, 'path_unchanged': sys.path == before}))
    """)
    self.assertEqual(result['found'], [False, False, False])
    self.assertTrue(result['path_unchanged'])

  def test_required_lookups(self):
    """
      Required lookups raise RavenNotFoundError, which is also a ModuleNotFoundError
      @ In, None
      @ Out, None
    """
    result = run_without_raven("""
      import json
      from HERON.src import _utils as hutils
      out = {}
      for name, func in [('raven', hutils.ensure_raven_on_path), ('teal', hutils.ensure_teal_on_path),
                         ('legacy', hutils._ensure_raven_importable)]:
        try:
          func()
          out[name] = 'no error'
        except hutils.RavenNotFoundError as e:
          out[name] = 'RavenNotFoundError' if isinstance(e, ModuleNotFoundError) else 'wrong base class'
      print(json.dumps(out))
    """)
    self.assertEqual(result, {'raven': 'RavenNotFoundError', 'teal': 'RavenNotFoundError',
                              'legacy': 'RavenNotFoundError'})


if __name__ == '__main__':
  unittest.main()

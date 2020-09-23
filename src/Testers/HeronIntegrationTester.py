
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED


import os
import sys

# get heron utilities
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(HERON_LOC)
import _utils as hutils
sys.path.pop()

# get RAVEN base testers
RAVEN_LOC = hutils.get_raven_loc()
TESTER_LOC = os.path.join(RAVEN_LOC, '..', 'scripts', 'TestHarness', 'testers')
sys.path.append(TESTER_LOC)
from RavenFramework import RavenFramework as RavenTester
sys.path.pop()

class HeronIntegration(RavenTester):
  """
    Defines testing mechanics for HERON integration tests; that is, tests that
    run the full HERON-RAVEN suite on a case.
  """

  # TODO extend get_valid_params?

  def __init__(self, name, param):
    """
      Constructor.
      @ In, name, str, name of test
      @ In, params, dict, test parameters
      @ Out, None
    """
    RavenTester.__init__(self, name, param)
    self.heron_driver = os.path.join(HERON_LOC, '..', 'heron')
    # NOTE: self.driver is RAVEN driver (e.g. /path/to/Driver.py)

  def get_command(self):
    """
      Gets the command line commands to run this test
      @ In, None
      @ Out, cmd, str, command to run
    """
    cmd = ''
    python = self._get_python_command()
    test_loc = os.path.abspath(self.specs['test_dir'])
    # HERON expects to be run in the dir of the input file currently, TODO fix this
    cmd += ' cd {loc} && '.format(loc=test_loc)
    # clear the subdirectory if it's present
    # FIXME it's not always Sweep_Runs; can we do git clean maybe?
    cmd += ' rm -rf Sweep_Runs_o/ ||: && '
    # run HERON first
    heron_inp = os.path.join(test_loc, self.specs['input'])
    cmd += f' {self.heron_driver} {heron_inp} && '
    # then run "outer.xml"
    raven_inp = os.path.abspath(os.path.join(os.path.dirname(heron_inp), 'outer.xml'))
    # TODO should this use raven_framework instead of "python Driver.py"?
    cmd += f' {python} {self.driver} {raven_inp}'
    return cmd


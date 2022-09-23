
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED


import os
import sys
import platform

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
  @staticmethod
  def get_valid_params():
    """
      Returns the valid parameters.
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = RavenTester.get_valid_params()
    params.add_param('kind', 'both', 'Run "both" HERON and RAVEN or "heron_only"')
    return params

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
    cmd, heron_inp = self.get_heron_command(cmd)
    if self.specs["kind"].lower() == "both":
      cmd += ' && '
      cmd = self.get_raven_command(cmd, heron_inp)
    elif self.specs["kind"].lower() == "heron_only":
      pass
    else:
      print("ERROR unknown HeronIntegration command kind", self.specs["kind"])
    return cmd

  def get_heron_command(self, cmd):
    """
      Generates command for running heron
      @ In, cmd, string
      @ Out, cmd, string, updated command
      @ Out, heron_inp
    """
    test_loc = os.path.abspath(self.specs['test_dir'])
    # HERON expects to be run in the dir of the input file currently, TODO fix this
    cmd += f' cd {test_loc} && '
    # clear the subdirectory if it's present
    cmd += ' rm -rf *_o/ && '
    # run HERON first
    heron_inp = os.path.join(test_loc, self.specs['input'])
    # Windows is a little different with bash scripts
    if platform.system() == 'Windows':
      cmd += ' bash.exe '
    cmd += f' {self.heron_driver} {heron_inp}'
    return cmd, heron_inp

  def get_raven_command(self, cmd, heron_inp):
    """
      Get command for running raven
      @ In, cmd, string
      @ In, heron_inp, path to heron input
      @ Out, cmd, string, updated command
    """
    python = self._get_python_command()
    raven_inp = os.path.abspath(os.path.join(os.path.dirname(heron_inp), 'outer.xml'))
    cmd += f' {python} {self.driver} {raven_inp}'
    return cmd

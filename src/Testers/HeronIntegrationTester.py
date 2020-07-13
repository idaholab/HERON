


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
    self.heron_driver = os.path.join(HERON_LOC, 'main.py')
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
    cmd += ' cd {loc}'.format(loc=test_loc)
    cmd += ' && '
    # clear the subdirectory if it's present
    cmd += ' rm -rf Sweep_Runs_o/ ||: && '
    # run HERON first
    heron_inp = os.path.join(test_loc, self.specs['input'])
    cmd += ' {py} {heron} {input}'.format(py=python,
                                          heron=self.heron_driver,
                                          input=heron_inp)
    # then run "outer.xml"
    ## TODO raven flags? So far I can't see it, but lets leave a spot
    raven_inp = os.path.abspath(os.path.join(os.path.dirname(heron_inp), 'outer.xml'))
    cmd += ' && ' # posix, only run second command if first one succeeds
    cmd += f' {python} {self.driver} {raven_inp}'
    # print('HERON command:', cmd)
    # print('\n\nDEBUGG dir:\n')
    # import pprint
    # pprint.pprint(self.__dir__())
    # print('\n\n')
    # print('DEBUGG specs:', self.specs)
    # print('\n\n')
    return cmd


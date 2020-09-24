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

class HeronPython(RavenTester):
  """
  Defines how to run HERON unit tests. These tests validate specific parts of
  the code.
  """

  def __init__(self, name, param):
    """
      Constructor.
      @ In, name, str, name of test
      @ In, params, dict, test parameters
      @ Out, None
    """
    RavenTester.__init__(self, name, param)

  def get_command(self):
    """
      Gets the command line commands to run this test
      @ In, None
      @ Out, cmd, str, command to run
    """
    test_loc = os.path.abspath(self.specs['test_dir'])
    input_file = self.specs['input']
    python = self._get_python_command()
    return f'cd {test_loc} && {python} {input_file}'
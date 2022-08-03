import os
import sys
import platform

# get heron utilities
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(HERON_LOC)
import _utils as hutils
from HeronIntegrationTester import HeronIntegration
sys.path.pop()

# get RAVEN base testers
RAVEN_LOC = hutils.get_raven_loc()
TESTER_LOC = os.path.join(RAVEN_LOC, '..', 'scripts', 'TestHarness', 'testers')
sys.path.append(TESTER_LOC)
from RavenFramework import RavenFramework as RavenTester
sys.path.pop()

class HeronMoped(HeronIntegration):
  """
    Defines testing mechanics for HERON integration tests for the MOPED workflow.
  """

  def __init__(self, name, param):
    """
      Constructor.
      @ In, name, str, name of test
      @ In, params, dict, test parameters
      @ Out, None
    """
    super().__init__(name, param)

  def get_command(self):
    """
      Gets the command line commands to run this test
      @ In, None
      @ Out, None
    """
    cmd = ''
    cmd,_ = self.get_heron_command(cmd)
    return cmd
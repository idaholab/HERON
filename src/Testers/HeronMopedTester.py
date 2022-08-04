import os
import sys
import platform

from HeronIntegrationTester import HeronIntegration

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
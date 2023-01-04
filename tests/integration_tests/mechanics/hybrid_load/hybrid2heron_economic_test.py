# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Test the capapility of loading economic information from HYBRID

Testing the python script (HERON/src/Hybrid2Heron/hybrid2heron_economic.py) that autoloads the needed economic information about a specific component or an element of the grid system from HYBRID text files to a HERON input xml file.
"""

import os
import sys

try:
  import HERON
  HYBRID_autoloader_path = HERON.__path__[0] + "/src/Hybrid2Heron/hybrid2heron_economic.py"
except ModuleNotFoundError:
  HYBRID_autoloader_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","..","src/Hybrid2Heron/hybrid2heron_economic.py")

# Execute the python script located at src/Hybrid2Heron
exec(open(HYBRID_autoloader_path).read())

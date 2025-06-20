
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
import sys
import os

# Make sure ravenframework can be imported
import HERON.src._utils as hutils
try:
  import ravenframework
except ModuleNotFoundError:
  framework_path = hutils.get_raven_loc()
  sys.path.append(framework_path)
  sys.path.append(os.path.abspath(os.path.join(framework_path, 'plugins')))
  sys.path.append(os.path.abspath(os.path.join(framework_path, 'scripts')))

try:
  import TEAL
except ModuleNotFoundError:
  framework_path = hutils.get_raven_loc()
  cashflow_path = os.path.abspath(os.path.join(hutils.get_cashflow_loc(raven_path=framework_path), '..'))
  sys.path.append(cashflow_path)

# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
try:
  import ravenframework
except ModuleNotFoundError:
  import sys
  from ._utils import get_raven_loc
  sys.path.append(get_raven_loc())

try:
  import dove
except ModuleNotFoundError:
  import sys
  from ._utils import get_plugin_loc
  sys.path.append(get_plugin_loc("DOVE"))

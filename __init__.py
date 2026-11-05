# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

# RAVEN discovers HERON's plugin entities (DispatchManager, DispatchPlot) by scanning this module.
# They subclass RAVEN classes, so they are only imported when RAVEN is available.
from HERON.src import _utils as _hutils
if _hutils.raven_available():
  from HERON.src import DispatchPlot
  from HERON.src import DispatchManager

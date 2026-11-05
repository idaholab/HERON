
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

# Make ravenframework and TEAL importable when available. HERON can be imported without them;
# modules that need RAVEN will fail at their own import with a clear RavenNotFoundError.
from . import _utils as hutils
hutils.ensure_raven_on_path(required=False)
hutils.ensure_teal_on_path(required=False)

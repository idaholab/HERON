# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Makes RAVEN and its plugins unimportable, so tests can check that HERON works without them even on
  machines where RAVEN is installed.
"""
import sys

# top-level modules treated as "RAVEN"; plugin_handler is RAVEN's plugin registry script
BLOCKED_MODULES = ('ravenframework', 'TEAL', 'FARM', 'plugin_handler')

class RavenImportBlocker:
  """
    Meta path finder that refuses to import RAVEN and its plugins. It raises the same error Python raises
    for a module that is not installed.
  """
  def find_spec(self, fullname, path=None, target=None):
    """
      Refuses to find blocked modules; defers everything else to the other finders
      @ In, fullname, str, fully qualified name of the module being imported
      @ In, path, list, optional, package search path (unused)
      @ In, target, module, optional, module being reloaded (unused)
      @ Out, spec, None, always None for modules that are not blocked
    """
    if fullname.split('.')[0] in BLOCKED_MODULES:
      raise ModuleNotFoundError(f"No module named '{fullname}' (blocked: testing without RAVEN)", name=fullname)
    return None

def block_raven():
  """
    Installs the blocker for the rest of the process. Fails if a blocked module was already imported,
    since the block could not take effect.
    @ In, None
    @ Out, None
  """
  loaded = sorted(name for name in sys.modules if name.split('.')[0] in BLOCKED_MODULES)
  if loaded:
    raise RuntimeError(f'Cannot block RAVEN: already imported {loaded}')
  if not any(isinstance(finder, RavenImportBlocker) for finder in sys.meta_path):
    sys.meta_path.insert(0, RavenImportBlocker())

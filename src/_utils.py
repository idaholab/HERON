
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  utilities for use within heron
"""
import sys
import importlib
import xml.etree.ElementTree as ET
import pickle
from functools import cache
from os import path


def get_heron_loc():
  """
    Return HERON location
    @ In, None
    @ Out, loc, string, absolute location of HERON
  """
  return path.abspath(path.join(__file__, '..', '..'))

def get_raven_loc():
  """
    Return RAVEN location
    hopefully this is read from heron/.ravenconfig.xml
    @ In, None
    @ Out, loc, string, absolute location of RAVEN
  """
  try:
    import ravenframework
    # Commented for now -- TODO: address these messages in the future
    # print("WARNING: get_raven_loc deprecated")
    # import traceback
    # traceback.print_stack()
    return path.dirname(ravenframework.__path__[0])
  except ModuleNotFoundError:
    pass
  config = path.abspath(path.join(path.dirname(__file__),'..','.ravenconfig.xml'))
  if not path.isfile(config):
    raise IOError(
        f'HERON config file not found at "{config}"! Has HERON been installed as a plugin in a RAVEN installation?'
    )
  loc = ET.parse(config).getroot().find('FrameworkLocation')
  assert loc is not None and loc.text is not None
  # The addition of ravenframework as an installable package requires
  # adding the raven directory to the PYTHONPATH instead of adding
  # ravenframework. We will expect '.ravenconfig.xml' to point to
  # raven/ravenframework always, so this is why we grab the parent dir.
  return path.abspath(path.dirname(loc.text))

def get_cashflow_loc(raven_path=None):
  """
    Get CashFlow (aka TEAL) location in installed RAVEN
    @ In, raven_path, string, optional, if given then start with this path
    @ Out, cf_loc, string, location of CashFlow
  """
  if raven_path is None:
    raven_path = get_raven_loc()
  plugin_handler_dir = path.join(raven_path, 'scripts')
  sys.path.append(plugin_handler_dir)
  sys.path.append(path.join(raven_path, 'scripts'))
  plugin_handler = importlib.import_module('plugin_handler')
  sys.path.pop()
  sys.path.pop()
  cf_loc = plugin_handler.getPluginLocation('TEAL')
  return cf_loc

def get_farm_loc(raven_path=None): # Added by Haoyu Wang, May 25, 2022
  """
    Get FARM location in installed RAVEN
    @ In, raven_path, string, optional, if given then start with this path
    @ Out, cf_loc, string, location of CashFlow
  """
  if raven_path is None:
    raven_path = get_raven_loc()
  plugin_handler_dir = path.join(raven_path, 'scripts')
  sys.path.append(plugin_handler_dir)
  plugin_handler = importlib.import_module('plugin_handler')
  sys.path.pop()
  farm_loc = plugin_handler.getPluginLocation('FARM')
  return farm_loc

def get_project_lifetime(case, components):
  """
    obtains the project lifetime
    @ In, case, HERON case, case
    @ In, components, list, HERON components
    @ Out, lifetime, int, project lifetime (usually years)
  """
  # load CashFlow
  try:
    from TEAL.src.main import getProjectLength
    from TEAL.src import CashFlows
  except (ImportError, ModuleNotFoundError) as e:
    loc = get_cashflow_loc()
    sys.path.append(loc)
    from TEAL.src.main import getProjectLength
    from TEAL.src import CashFlows
    sys.path.pop()
  econ_comps = list(comp.get_economics() for comp in components)
  econ_params = case.get_econ(econ_comps)
  econ_settings = CashFlows.GlobalSettings()
  econ_settings.setParams(econ_params)
  return getProjectLength(econ_settings, econ_comps)

class RavenNotFoundError(ModuleNotFoundError):
  """
    Raised when a feature that requires RAVEN (or a RAVEN plugin such as TEAL) is used but it cannot be
    imported. Subclasses ModuleNotFoundError so existing "except ModuleNotFoundError" handlers still apply.
  """

_RAVEN_INSTALL_HELP = ('Install RAVEN (e.g. "pip install raven-framework") or register HERON as a plugin of a '
                       'RAVEN source checkout ("python <raven>/scripts/install_plugins.py -s <path to HERON>").')

def _is_missing(err, module):
  """
    Checks whether an import error means the given top-level module itself is missing, as opposed to
    the module being present but failing on one of its own imports.
    @ In, err, ModuleNotFoundError, error raised by the import
    @ In, module, str, name of the top-level module that was imported
    @ Out, missing, bool, True if the module itself could not be found
  """
  return err.name is not None and err.name.split('.')[0] == module

def _add_to_path(entries):
  """
    Appends entries to sys.path, skipping any already present
    @ In, entries, list(str), paths to add
    @ Out, added, list(str), paths that were actually added
  """
  added = []
  for entry in entries:
    entry = path.abspath(path.expanduser(entry))
    if entry not in sys.path:
      sys.path.append(entry)
      added.append(entry)
  return added

def ensure_raven_on_path(required=True):
  """
    Makes ravenframework importable, adding a RAVEN checkout (found via get_raven_loc) to sys.path if needed.
    @ In, required, bool, optional, if True raise when RAVEN cannot be imported; else return False
    @ Out, found, bool, True if ravenframework is importable
  """
  try:
    import ravenframework
    return True
  except ModuleNotFoundError as e:
    if not _is_missing(e, 'ravenframework'):
      raise # RAVEN is present but broken (e.g. a missing library); don't hide that

  # not importable as installed; look for a RAVEN checkout that HERON is registered with
  try:
    raven_loc = get_raven_loc()
  except (IOError, OSError) as e:
    if required:
      raise RavenNotFoundError(f'RAVEN is required but could not be found. {_RAVEN_INSTALL_HELP}',
                               name='ravenframework') from e
    return False

  added = _add_to_path([raven_loc, path.join(raven_loc, 'plugins'), path.join(raven_loc, 'scripts')])
  importlib.invalidate_caches()

  try:
    import ravenframework
    return True
  except ModuleNotFoundError as e:
    if not _is_missing(e, 'ravenframework'):
      raise
    # the configured location doesn't hold a usable RAVEN; undo our changes to sys.path
    for entry in added:
      sys.path.remove(entry)
    if required:
      raise RavenNotFoundError(f'RAVEN is required but could not be imported from "{raven_loc}". '
                               f'{_RAVEN_INSTALL_HELP}', name='ravenframework') from e
    return False

def ensure_teal_on_path(required=True):
  """
    Makes the TEAL plugin importable, locating it through RAVEN's plugin registry if needed.
    @ In, required, bool, optional, if True raise when TEAL cannot be imported; else return False
    @ Out, found, bool, True if TEAL is importable
  """
  try:
    import TEAL
    return True
  except ModuleNotFoundError as e:
    if not _is_missing(e, 'TEAL'):
      raise

  teal_loc = None
  if ensure_raven_on_path(required=False):
    try:
      teal_loc = get_cashflow_loc()
    except (IOError, OSError, ModuleNotFoundError):
      teal_loc = None # no plugin registry available

  if teal_loc is None:
    if required:
      raise RavenNotFoundError(f'The TEAL plugin is required but could not be found. {_RAVEN_INSTALL_HELP}',
                               name='TEAL')
    return False

  # TEAL is imported as a package, so its parent directory goes on the path
  added = _add_to_path([path.join(teal_loc, '..')])
  importlib.invalidate_caches()

  try:
    import TEAL
    return True
  except ModuleNotFoundError as e:
    if not _is_missing(e, 'TEAL'):
      raise
    for entry in added:
      sys.path.remove(entry)
    if required:
      raise RavenNotFoundError(f'The TEAL plugin is required but could not be imported from "{teal_loc}". '
                               f'{_RAVEN_INSTALL_HELP}', name='TEAL') from e
    return False

@cache
def raven_available():
  """
    Checks (once per process) whether RAVEN can be imported, setting up sys.path if needed
    @ In, None
    @ Out, available, bool, True if ravenframework is importable
  """
  return ensure_raven_on_path(required=False)

def _ensure_raven_importable():
  """
    Adds RAVEN to the path if ravenframework cannot be imported, so serialized ROMs can be loaded
    @ In, None
    @ Out, None
  """
  ensure_raven_on_path(required=True)

@cache
def get_synthhist_structure(fpath):
  """
    Extracts synthetic history info from ROM (currently ARMA ROM)
    @ In, fpath, str, full absolute path to serialized ROM
    @ Out, structure, dict, derived structure from reading ROM XML
  """
  # TODO could this be a function of the ROM itself?
  # TODO or could we interrogate the ROM directly instead of the XML?
  #If ravenframework not in path, need to add, otherwise loading rom will fail
  _ensure_raven_importable()
  rom = pickle.load(open(fpath, 'rb'))

  structure = {}
  meta = rom.writeXML().getRoot()
  # interpolation information
  interp_node = meta.find('InterpolatedMultiyearROM')
  if interp_node:
    macro_id = interp_node.find('MacroParameterID').text.strip()
    structure['macro'] = {
      'id': macro_id,
      'num': int(interp_node.find('MacroSteps').text),
      'first': int(interp_node.find('MacroFirstStep').text),
      'last': int(interp_node.find('MacroLastStep').text),
    }
    macro_nodes = meta.findall('MacroStepROM')
  else:
    macro_nodes = [meta]

  # cluster information
  structure['clusters'] = {}
  for macro in macro_nodes:
    if interp_node:
      macro_index = int(macro.attrib[macro_id])
    else:
      macro_index = 0
    clusters_info = [] # data dict for each macro step
    structure['clusters'][macro_index] = clusters_info
    cluster_nodes = macro.findall('ClusterROM')
    if cluster_nodes:
      for node in cluster_nodes:
        info = {
          'id': int(node.attrib['cluster']),
          'represents': node.find('segments_represented').text.split(','),
          'indices': list(int(x) for x in node.find('indices').text.split(','))
        }
        clusters_info.append(info)
  # segment information
  # -> TODO
  structure['segments'] = {}
  return structure

def sample_synthhist(fpath, eval_mode='clustered'):
  """
    Evaluates a serialized synthetic history ROM once, quietly, as HERON's dispatch would see it
    @ In, fpath, str, full absolute path to serialized ROM
    @ In, eval_mode, str, optional, ROM cluster evaluation mode ('clustered' or 'full')
    @ Out, realization, dict, one sampled realization, {variable name: np.array}, including the
      pivot (time) and macro (year) variables, '_ROM_Cluster' if clustered, and '_indexMap'
  """
  import io
  import contextlib
  _ensure_raven_importable()
  from ravenframework.ROMExternal import ROMLoader
  from HERON.src.xml_utils import new_node
  runner = ROMLoader(fpath)
  node = new_node('ROM', attrib={'name': 'SyntheticHistory', 'subType': 'pickledRom'})
  node.append(new_node('clusterEvalMode', text=eval_mode))
  # RAVEN prints a message per evaluated segment; keep those out of the caller's output
  with contextlib.redirect_stdout(io.StringIO()):
    runner.setAdditionalParams([node])
    realization = runner.evaluate({'scaling': [1]})[0]
  return realization

def get_csv_structure(fpath, macro_var, micro_var):
  """
    Returns CSV structure in a way RAVEN & HERON understand
    @ In, fpath, str, file path to CSV file
    @ In, macro_var, str, Macro Variable name - typically 'Year'
    @ In, micro_var, str, Micro Variable name - typically 'Time'
    @ Out, structure, dict, Nested structure of the CSV dataframe.
  """
  import pandas as pd #Note that this cannot be imported at the start of this
  # file since _utils.py is used in heron script outside of the raven environment
  # to find the environment.
  data = pd.read_csv(fpath)
  structure = {}

  structure['num_samples'] = data['RAVEN_sample_ID'].unique().size

  if macro_var in data.columns:
    macro_steps = pd.unique(data[macro_var].values)
    structure['macro'] = {
      'id': macro_var,
      'num': len(macro_steps) + 1,
      'first': min(macro_steps),
      'last': max(macro_steps)
    }
  else:
    data[macro_var] = 0

  structure['clusters'] = {}
  for macro_step, df in data.groupby(macro_var):
    structure['clusters'][macro_step] = [{
      'id': 0,
      'indices': [0, len(df[micro_var].values)],
      'represents': ['0']
    }]

  structure['segments'] = {}  # TODO
  return structure


if __name__ == '__main__':
  try:
    action = sys.argv[1]
  except IndexError:
    print('Please provide the method to run as the first argument.')
  if action == 'get_raven_loc':
    print(get_raven_loc())
  elif action == 'get_cashflow_loc':
    print(get_cashflow_loc())
  else:
    raise IOError(f'Unrecognized action: {action}')


# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  utilities for use within heron
"""
import sys
import importlib
import xml.etree.ElementTree as ET
import warnings
import pickle
try:
  from functools import cache
except ImportError:
  from functools import lru_cache
  def cache(user_function):
     """
       use lru_cache for older versions of python
       @ In, user_function, function
       @ Out, user_function, function that caches values
     """
     return lru_cache(maxsize=None)(user_function)
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

def get_all_resources(components):
  """
    Provides a set of all resources used among all components
    @ In, components, list, HERON component objects
    @ Out, resources, list, resources used in case
  """
  res = set()
  for comp in components:
    res.update(comp.get_resources())
  return res

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

@cache
def get_synthhist_structure(fpath):
  """
    Extracts synthetic history info from ROM (currently ARMA ROM)
    @ In, fpath, str, full absolute path to serialized ROM
    @ Out, structure, dict, derived structure from reading ROM XML
  """
  # TODO could this be a function of the ROM itself?
  # TODO or could we interrogate the ROM directly instead of the XML?
  try:
    import ravenframework
  except ModuleNotFoundError:
    #If ravenframework not in path, need to add, otherwise loading rom will fail
    raven_path = hutils.get_raven_loc()
    sys.path.append(os.path.expanduser(raven_path))
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
    raise IOError('Unrecognized action: "{}"'.format(action))


# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  utilities for use within heron
"""
import os
import sys
import importlib
import xml.etree.ElementTree as ET

def get_heron_loc():
  """
    Return HERON location
    @ In, None
    @ Out, loc, string, absolute location of HERON
  """
  return os.path.abspath(os.path.join(__file__, '..', '..'))

def get_raven_loc():
  """
    Return RAVEN location
    hopefully this is read from heron/.ravenconfig.xml
    @ In, None
    @ Out, loc, string, absolute location of RAVEN
  """
  config = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','.ravenconfig.xml'))
  if not os.path.isfile(config):
    raise IOError('HERON config file not found at "{}"! Has HERON been installed as a plugin in a RAVEN installation?'
                  .format(config))
  loc = ET.parse(config).getroot().find('FrameworkLocation').text
  return loc

def get_cashflow_loc(raven_path=None):
  """
    Get CashFlow (aka TEAL) location in installed RAVEN
    @ In, raven_path, string, optional, if given then start with this path
    @ Out, cf_loc, string, location of CashFlow
  """
  if raven_path is None:
    raven_path = get_raven_loc()
  plugin_handler_dir = os.path.join(raven_path, '..', 'scripts')
  sys.path.append(plugin_handler_dir)
  plugin_handler = importlib.import_module('plugin_handler')
  sys.path.pop()
  cf_loc = plugin_handler.getPluginLocation('TEAL')
  return cf_loc

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

def get_synthhist_structure(fpath):
  """
    Extracts synthetic history info from ROM (currently ARMA ROM)
    @ In, fpath, str, full absolute path to serialized ROM
    @ Out, structure, dict, derived structure from reading ROM XML
  """
  # TODO could this be a function of the ROM itself?
  # TODO or could we interrogate the ROM directly instead of the XML?
  raven_loc = get_raven_loc()
  scripts_path = os.path.join(raven_loc, '..', 'scripts')
  sys.path.append(scripts_path)
  from externalROMloader import ravenROMexternal as ravenROM
  rom = ravenROM(fpath, raven_loc).rom
  meta = rom.writeXML().getRoot()
  structure = {}
  # interpolation information
  interp_node = meta.find('InterpolatedMultiyearROM')
  if interp_node:
    macro_id = interp_node.find('MacroParameterID').text.strip()
    structure['macro'] = {'id': macro_id,
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
        info = {'id': int(node.attrib['cluster']),
                'represents': node.find('segments_represented').text.split(','),
                'indices': list(int(x) for x in node.find('indices').text.split(','))
               }
        clusters_info.append(info)
  # segment information
  # -> TODO
  structure['segments'] = {}
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

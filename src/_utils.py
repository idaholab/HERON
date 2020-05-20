"""
  utilities for use within heron
"""
import os
import sys
import importlib
import xml.etree.ElementTree as ET

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
    Get CashFlow location in installed RAVEN
    @ In, raven_path, string, optional, if given then start with this path
    @ Out, cf_loc, string, location of CashFlow
  """
  if raven_path is None:
    raven_path = get_raven_loc()
  plugin_handler_dir = os.path.join(raven_path, '..', 'scripts')
  sys.path.append(plugin_handler_dir)
  plugin_handler = importlib.import_module('plugin_handler')
  sys.path.pop()
  cf_loc = plugin_handler.getPluginLocation('CashFlow')
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

if __name__ == '__main__':
  action = sys.argv[1]
  if action == 'get_raven_loc':
    print(get_raven_loc())
  elif action == 'get_cashflow_loc':
    print(get_cashflow_loc())
  else:
    raise IOError('Unrecognized action: "{}"'.format(action))


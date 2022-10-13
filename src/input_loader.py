
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Parses an input file, returning the objects therein.
"""
import sys
import xml.etree.ElementTree as ET

from . import Cases
from . import Components
from . import Placeholders
from . import ValuedParams

from . import _utils as hutils
try:
  import ravenframework
except ModuleNotFoundError:
  raven_path = hutils.get_raven_loc()
  sys.path.append(raven_path)
from ravenframework.utils import xmlUtils



def load(name):
  """
    Loads file into XML format
    @ In, name, str, name of file to read from
    @ Out, xml, ET.element, input as xml
  """
  xml, _ = xmlUtils.loadToTree(name)
  return xml

def parse(xml, loc, messageHandler):
  """
    Read XML and produce object instances
    @ In, xml, ET.element, to read from
    @ In, loc, str, location of input xml
    @ In, messageHandler, RAVEN messageHandler, handler of messages
    @ Out, parse, dict, constructed entities
  """
  # storage
  case = None      # TODO
  components = []  # grid components
  sources = []     # source/demand signals, functions

  # intentionally read case first
  case_node = xml.find('Case')
  if case_node is None:
    raise IOError('<Case> node is missing from HERON input file!')
  case = Cases.Case(loc, messageHandler=messageHandler)
  case.read_input(case_node)

  # components
  components_node = xml.find('Components')
  if components_node is None:
    raise IOError('<Components> node is missing from HERON input file!')
  for comp_xml in components_node:
    comp = Components.Component(messageHandler=messageHandler)
    # check parsing
    comp.read_input(comp_xml, case.get_mode())
    components.append(comp)
  if not components:
    raise IOError('No <Component> nodes were found in the <Components> section!')

  sources_node = xml.find('DataGenerators')
  if sources_node is None:
    raise IOError('<DataGenerators> node is missing from HERON input file!')
  # TODO only load the generators we need?? it'd be a good idea ...
  for sub_xml in sources_node:
    typ = sub_xml.tag
    if typ == 'CSV':
      new = Placeholders.CSV(loc=loc, messageHandler=messageHandler)
    elif typ == 'ARMA':
      new = Placeholders.ARMA(loc=loc, messageHandler=messageHandler)
    elif typ == 'Function':
      new = Placeholders.Function(loc=loc, messageHandler=messageHandler)
    elif typ == 'ROM':
      new = Placeholders.ROM(loc=loc, messageHandler=messageHandler)
    else:
      raise IOError('Unrecognized DataGenerator: "{}"'.format(sub_xml.tag))
    new.read_input(sub_xml)
    sources.append(new)

  # now go back through and link up stuff
  for comp in components:
    found = {}
    for obj, info in comp.get_crossrefs().items():
      found[obj] = {}
      for attr, info in info.items():
        kind, name = info.get_source()
        # if not looking for a DataGenerator placeholder, then nothing more to do
        # if using "activity", also nothing to do
        if kind not in ['Function', 'ARMA', 'ROM', 'CSV']:
          continue
        # find it
        for source in sources:
          if source.is_type(kind) and source.name == name:
            found[obj][attr] = source
            break
        else:
          raise IOError(f'Requested source "{name}" for component "{comp.name}" was not found!')
    comp.set_crossrefs(found)

  # then do pre-writing initialization
  case.initialize(components, sources)

  return {'case': case,
          'components': components,
          'sources': sources,
         }

if __name__ == "__main__":
  inp = load(sys.argv[1])
  obj = parse(inp, None)
  for k, v in obj.items():
    print('  ', k, v)

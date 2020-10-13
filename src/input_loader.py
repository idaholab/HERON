
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Parses an input file, returning the objects therein.
"""
import os
import sys

import Cases
import Components
import Placeholders
import time
import xml.etree.ElementTree as ET

import _utils as hutils
raven_path = hutils.get_raven_loc()
sys.path.append(raven_path)
from utils import xmlUtils
sys.path.pop()


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
  case = Cases.Case(loc, messageHandler=messageHandler)
  case.read_input(case_node)

  # read everything else
  for section in xml:
    if section.tag == 'Components':
      for comp_xml in section:
        #print(type(Components.Component()))
        #time.sleep(2000)
        comp = Components.Component(messageHandler=messageHandler)
        # check parsing
        comp.read_input(comp_xml, case.get_mode())
        components.append(comp)

    elif section.tag == 'DataGenerators':
      # TODO only load the generators we need?? it'd be a good idea ...
      for sub_xml in section:
        typ = sub_xml.tag
        if typ == 'CSV':
          raise NotImplementedError('Not taking histories from CSV yet. If needed, let me know.')
          new = Placeholders.CSV(messageHandler=messageHandler)
        elif typ == 'ARMA':
          new = Placeholders.ARMA(loc=loc, messageHandler=messageHandler)
          #print("THIS IS INPUT LOADER")
        elif typ == 'Function':
          new = Placeholders.Function(loc=loc, messageHandler=messageHandler)
        else:
          raise IOError('Unrecognized DataGenerator: "{}"'.format(sub_xml.tag))
        new.read_input(sub_xml)
        sources.append(new)

  # now go back through and link up stuff
  # TODO move to case.initialize?
  for comp in components:
    found = {}
    for interaction, i_info in comp.get_crossrefs().items():
      found[interaction] = {}
      for attr, info in i_info.items():
        typ, name = info.get_source()
        # if not looking for a DataGenerator placeholder, then nothing more to do
        if typ not in ['Function', 'ARMA']:
          continue
        # find it
        for source in sources:
          if source.is_type(typ) and source.name == name:
            found[interaction][attr] = source
            break
        else:
          raise IOError('Requested source "{}" for component "{}" was not found!'.format(name, comp.name))
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

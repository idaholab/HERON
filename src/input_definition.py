#
# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Load HERON input schematic and emit a WASP-formatted input definition
  for use with EDDI-formatted input in the NEAMS Workbench
"""
import sys
import xml.etree.ElementTree as ET

from . import Cases
from . import Components
from . import Placeholders

from . import _utils as hutils
try:
  import ravenframework
except ModuleNotFoundError:
  raven_path = hutils.get_raven_loc()
  sys.path.append(raven_path)
from ravenframework.utils import xmlUtils
from ravenframework.utils.InputData import Quantity

def indent(level):
    """
      Obtain an indent prefix whitespace string.
      @ In, level, int, the number of indentation levels to use.
      @ Out, indent, str, the resultant indentation
    """
    return " "*level*2

def emitValEnumDefinition(cls, level):
    """
       Emit the value enumeration restriction
       @ In, cls, ParameterInput class, component definition
       @ In, level, int, the number of indentation levels to use for this component.
       @ Out, None
    """
    enum = ' '.join(cls.enumList)
    print(indent(level),"ValEnums[",enum,"]")

def emitDefinition(cls, level=0, occurs=None):
  """
    Generates the input definition information for this node.
    @ In, cls, ParameterInput class, component definition
    @ In, level, int, the number of indentation levels to use for this component.
    @ In, occurs, tuple(min,max), the occurrence restriction for this component. Max can be 'NoLimit'
    @ Out, None
  """
  print(indent(level), cls.getName(), "{")
  if cls.description:
    print(indent(level+1),"Description=\""+cls.description.replace("\n"," ") \
                                                           .replace("               ", " ") \
                                                           .replace("\"","'") \
                                                           .replace("         ","")+"\"")
  templateName = "'element'"
  if occurs is not None:
    # None indicates the default of 0 and NoLimit and not need to be output
    print(indent(level), "MinOccurs=",occurs[0]," MaxOccurs=",occurs[1])
  if cls.subs:
      if cls.subOrder is not None:
        subList = cls.subOrder
      else:
        subList = [(sub, Quantity.zero_to_infinity) for sub in cls.subs]
      for sub, quantity in subList:
        subOccurs = None
        if cls.subOrder is not None:
          if quantity == Quantity.zero_to_one:
            occurs = ('0','1')
          elif quantity == Quantity.zero_to_infinity:
            subOccurs = None # Default is ('0','NoLimit')
          elif quantity == Quantity.one:
            subOccurs = ('1','1')
          elif quantity == Quantity.one_to_infinity:
            subOccurs = ('1','NoLimit')
          else:
            print("ERROR unexpected quantity ",quantity)
        emitDefinition(sub, level+1, subOccurs)
  else:

    if hasattr(cls, 'enumList') and cls.enumList is not None:
        print(indent(level+2),"value{MinOccurs=1 MaxOccurs=1")
        emitValEnumDefinition(cls, level+3)
        print(indent(level+2),"}")
    if cls.contentType is not None:
      xmltype = cls.contentType.getXMLType()
      # complex types (those not a part of xsd) are captured by
      # other constraints. We don't output string as this is the default
      isXSD = xmltype.startswith("xsd:")

      print(indent(level+2),"value{")
      if isXSD and xmltype != "xsd:string":
        print(indent(level+3), "ValType=", {"double":"Real", "integer":"Int"}[xmltype[4:]])

      if not isXSD:
        emitValEnumDefinition(cls.contentType, level+3)
      print(indent(level+3),"MinOccurs=0 MaxOccurs=NoLimit")
      print(indent(level+2),"}")
  #generate attributes and determine if it has a 'name' attribute
  hasName = False
  for parameter in cls.parameters:
    parameterData = cls.parameters[parameter]
    print(indent(level+1), parameter, "{")
    dataType = parameterData["type"]
    isRequired = parameterData["required"]
    # alias name to first index of parent element
    if parameter == "name":
      hasName = True
      print(indent(level+2), "InputAliases[\"_0\"]")
    elif hasattr(dataType, 'enumList'):
      print(indent(level+2),"value{MinOccurs=1 MaxOccurs=1")
      emitValEnumDefinition(dataType, level+3)
      print(indent(level+2),"}")

    if isRequired:
      print(indent(level+2), 'MinOccurs=1 MaxOccurs=1')
    else:
      # attributes can only occur maximally once
      print(indent(level+3), "MaxOccurs=1")
    print(indent(level+2), "InputTmpl='attribute'")
    print(indent(level+1), "} % end of attribute ", parameter)
  if cls.subs:
    # if element has a 'name' attribute we alias name to first value
    # else no name AND sub elements we must highlight 'value' nodes are UNKNOWN
    if hasName == False:
      print(indent(level+1), "InputTmpl='element'")
      print(indent(level+1),"value(UNKNOWN){}")
    else:
      print(indent(level+1), "InputTmpl='named-element'")
  else:
    print(indent(level+1), "InputTmpl='leaf-element'")
  print(indent(level), '} % end of element ', cls.getName())

def print_input_definition():
  """
    Obtain object input specifications and print input definition to stdout
    @ In, None
    @ Out, None
  """
  print("%-START-SON-DEFINITION-%")
  print("% SON-DEFINITION is defined by rules documented at https://code.ornl.gov/neams-workbench/wasp/-/blob/master/wasphive/README.md")

  emitDefinition(Cases.Case("~").get_input_specs(), level=0, occurs=None)
  print("Components{ MinOccurs=1 InputTmpl=element")
  emitDefinition(Components.Component(loc="~").get_input_specs(), level=0, occurs=None)
  print("}")
  # deal with DataGenerators
  print("DataGenerators{ MinOccurs=1 InputTmpl=element")
  for obj in [Placeholders.CSV(loc="~"),Placeholders.ARMA(loc="~"), Placeholders.Function(loc="~"), Placeholders.ROM(loc="~")]:
    emitDefinition(obj.get_input_specs(), level=1, occurs=None)
  print("}")
  # Ensure value nodes cannot exist at the root of the document
  print(indent(0),"value(UNKNOWN){}")
  print("%-END-SON-DEFINITION-%")


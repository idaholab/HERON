# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  DataObject snippets

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
import xml.etree.ElementTree as ET

from ..xml_utils import find_node
from ..decorators import listproperty
from .base import RavenSnippet


class DataObject(RavenSnippet):
  """ DataObject base class """
  snippet_class = "DataObjects"

  @listproperty
  def inputs(self) -> list[str]:
    """
    Getter for inputs list
    @ In, None
    @ Out, inputs, list[str], list of input variables
    """
    node = self.find("Input")
    return getattr(node, "text", [])

  @inputs.setter
  def inputs(self, value: list[str]) -> None:
    """
    Setter for inputs list
    @ In, value, list[str], list of input variables
    @ Out, None
    """
    find_node(self, "Input").text = value

  @listproperty
  def outputs(self) -> list[str]:
    """
    Getter for outputs list
    @ In, None
    @ Out, outputs, list[str], list of output variables
    """
    node = self.find("Output")
    return getattr(node, "text", [])

  @outputs.setter
  def outputs(self, value: list[str]) -> None:
    """
    Setter for outputs list
    @ In, value, list[str], list of output variables
    @ Out, None
    """
    find_node(self, "Output").text = value


class PointSet(DataObject):
  """ PointSet snippet """
  tag = "PointSet"


class HistorySet(DataObject):
  """ HistorySet snippet """
  tag = "HistorySet"


class DataSet(DataObject):
  """ DataSet snippet """
  tag = "DataSet"

  def add_index(self, index_var: str, variables: str | list[str]) -> None:
    """
    Add an index node to the snippet XML
    @ In, index_var, str, name of the index variable
    @ In, variables, str | list[str], names of the variable(s) indexed by the index_var
    @ Out, None
    """
    # Force to be a list
    if not isinstance(variables, list):
      variables = [str(variables)]

    # There shouldn't be any reason to have duplicate index nodes, so only add the indicated index variable
    index_node = self.find(f"Index[@var='{index_var}']")
    if index_node is None:
      ET.SubElement(self, "Index", {"var": index_var}).text = variables
    else:
      # If there are any variables provided that aren't in the existing index node's text, add them here.
      for var in variables:
        if var not in index_node.text:
          index_node.text.append(var)

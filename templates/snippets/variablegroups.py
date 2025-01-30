# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Variable group snippet class

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
import xml.etree.ElementTree as ET

from .base import RavenSnippet
from ..decorators import listproperty


class VariableGroup(RavenSnippet):
  """ A group of variable names """
  snippet_class = "VariableGroups"
  tag = "Group"

  @classmethod
  def from_xml(cls, node: ET.Element) -> "VariableGroup":
    """
    Alternate construction from XML node
    @ In, node, ET.Element, the XML node
    @ Out, vargroup, VariableGroup, the corresponding variable group object
    """
    vargroup = cls(node.get("name"))
    if node.text:
      var_names = [varname.strip() for varname in node.text.split(",")]
      vargroup.variables = var_names
    return vargroup

  @listproperty
  def variables(self) -> list[str]:
    """
    Getter for list of variables in group
    @ In, None
    @ Out, variables, list of variables in group
    """
    return self.text

  @variables.setter
  def variables(self, value: list[str]) -> None:
    """
    Setter for variables list
    @ In, value, list[str], list of variable names
    @ Out, None    """
    self.text = value

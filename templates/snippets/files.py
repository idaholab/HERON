# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  File snippets

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
import xml.etree.ElementTree as ET

from .base import RavenSnippet


class File(RavenSnippet):
  """ Snippet class for input files """
  snippet_class = "Files"
  tag = "Input"

  def to_assembler_node(self, tag: str) -> ET.Element:
    """
    Refer to the snippet with an assembler node representation
    @ In, tag, str, the tag of the assembler node to use
    @ Out, node, ET.Element, the assembler node
    """
    node = super().to_assembler_node(tag)
    # Type is set as a node attribute and is not the main node tag, as is assumed in the default implementation.
    # The "type" attribute should be an empty string if not set.
    node.set("type", self.type or "")
    return node

  @property
  def type(self) -> str | None:
    """
    type getter
    @ In, None,
    @ Out, type, str | None, the file type
    """
    return self.get("type", None)

  @type.setter
  def type(self, value: str) -> None:
    """
    type setter
    @ In, value, str, the type value to set
    @ Out, None
    """
    self.set("type", value)

  @property
  def path(self) -> str | None:
    """
    file path getter
    @ In, None,
    @ Out, path, str | None, the file path
    """
    return self.text

  @path.setter
  def path(self, value: str) -> None:
    """
    file path setter
    @ In, value, str, the file path to set
    @ Out, None
    """
    self.text = value

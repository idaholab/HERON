"""
Mocks for unit testing RavenSnippet subclasses
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import xml.etree.ElementTree as ET


class MockSnippet:
  """ Mock for the RavenSnippet classes """
  name = None
  snippet_class = None
  tag = None

  def __init__(self, name: str | None = None, snippet_class: str | None = None, tag: str | None = None) -> None:
    """
    Constructor
    @ In, name, str, optional, the snippet object's name
    @ In, snippet_class, str, optional, the snippet class name
    @ In, tag, str, optional, the XML tag to use
    @ Out, None
    """
    self.name = name or "some_name"
    self.snippet_class = snippet_class or "cls"
    self.tag = tag or "some_tag"

  def to_assembler_node(self, tag: str) -> ET.Element:
    """
    Make a RAVEN assembler XML node from the snippet object
    @ In, tag, str, the XML tag
    @ Out, node, ET.Element, the assembler node
    """
    node = ET.Element(tag, attrib={"class": self.snippet_class, "type": self.tag})
    node.text = self.name
    return node

  def __repr__(self) -> str:
    """
    String representation the mock snippet object using just the snippet name
    @ In, None
    @ Out, name, str, the snippet name
    """
    return self.name

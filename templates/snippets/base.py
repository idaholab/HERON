# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Base RavenSnippet class

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
from typing import Any
import xml.etree.ElementTree as ET

from ..xml_utils import merge_trees

class RavenSnippet(ET.Element):
  """
  RavenSnippet class objects describe one contiguous snippet of RAVEN XML, inheriting from the
  xml.etree.ElementTree.Element class. This base class contains methods for quickly building subtrees
  and set and access common RAVEN node attributes.
  """
  tag = None  # XML tag associated with the snippet class
  snippet_class = None  # class of Entity described by the snippet (e.g. Models, Optimizers, DataObjects, etc.)
  subtype = None  # subtype of the snippet entity, does not need to be defined for all snippets

  @classmethod
  def from_xml(cls, node: ET.Element, **kwargs) -> "RavenSnippet":
    """
    Alternate constructor which instantiates a new RavenSnippet object from an existing XML node
    @ In, node, ET.Element, the template node
    @ In, kwargs, dict, keyword arguments
    @ Out, snippet, RavenSnippet, the new snippet
    """
    # Instantiate the snippet class and copy over attribs and text
    snippet = cls()
    snippet.attrib |= node.attrib
    snippet.text = node.text

    # Merge the node's subtree into the snippet's subtree. Whether or not to overwrite equal nodes (overwrite=True),
    # include both nodes (overwrite=False), and whether or not matching nodes must have matching attributes
    # (match_attrib) can be contolled via keyword arguments.
    merge_kwargs = {k: kwargs[k] for k in kwargs.keys() & {"overwrite", "match_attrib", "match_text"}}
    snippet = merge_trees(snippet, node, **merge_kwargs)

    return snippet

  def __init__(self, name: str | None = None, subelements: dict[str, Any] | None = None) -> None:
    """
    Constructor
    @ In, name, str, optional, the name of the entity
    @ In, subelements, dict[str, Any], optional, keyword settings which are added as XML child nodes
    @ Out, None
    """
    super().__init__(self.tag)

    # Update node attributes with provided values
    # Arguments "name", "class_name", and "subtype_name" help to alias the problematic "class" attribute name and
    # provide an easy interface to set the common attributes "name" and "subType".
    if name is not None:
      self.name = name
    if self.subtype is not None:
      self.attrib["subType"] = self.subtype

    if subelements:
      self.add_subelements(subelements)

  def __repr__(self) -> str:
    """
    Make a string representation of the snippet. If the "name" attribute is defined, return that. Otherwise, fall back
    to the ET.Element implementation.
    @ In, None
    @ Out, repr, str, the string representation of the object
    """
    if name := self.name:
      return name
    return super().__repr__()

  @property
  def name(self) -> str | None:
    """
    Name attribute getter
    @ In, None
    @ Out, name, str | None, the name
    """
    return self.get("name", None)

  @name.setter
  def name(self, value: str) -> None:
    """
    Name attribute setter
    @ In, value, str, the name to set
    @ Out, None
    """
    self.set("name", value)

  def to_assembler_node(self, tag: str) -> ET.Element:
    """
    Creates an assembler node from the snippet, if possible. The "class" attribute must be defined.
    @ In, tag, str, assembler node tag
    """
    if not (self.snippet_class and self.name):
      raise ValueError("The RavenSnippet object cannot be expressed as an Assembler node! The object must have "
                       "'name' and 'class' attributes defined to create an Assembler node. Current values: "
                       f"class='{self.snippet_class}', name='{self.name}'.")

    node = ET.Element(tag)
    node.attrib["class"] = self.snippet_class
    node.attrib["type"] = self.tag
    node.text = self.name

    return node

  # Subtree building utilities
  def add_subelements(self, subelements: dict[str, Any] | None = None, **kwargs) -> None:
    """
    Add subelements by either providing a dict or keyword arguments.
    @ In, subelements, dict[str, Any], optional, dict with new key-value settings pairs
    @ In, kwargs, dict, optional, new settings provided as keyword arguments
    @ Out, None
    """
    parent = kwargs.pop("parent", self)
    all_subs = kwargs | (subelements if subelements else {})
    for tag, value in all_subs.items():
      self._add_subelement(parent, tag, value)

  def _add_subelement(self, parent: ET.Element, tag: str, value: Any) -> None:
    """
    Recursively build out subtree. Recurse over dicts, set child node text to string or numeric values,
    or form comma separated lists for other iterative data types (list, numpy array, tuple, set, etc.).
    @ In, parent, ET.Element, the parent node to append to
    @ In, tag, str, the tag of the child node
    @ In, value, Any, the value of the child node
    """
    # If the value inherits from ET.Element, we can append the value to the parent directly.
    if isinstance(value, ET.Element):
      parent.append(value)
    # If the value happens to be another entity, it has its own to_xml method. Use that instead of manually
    # using the tag input to create the child node.
    elif isinstance(value, RavenSnippet):
      # has a to_xml method
      child = value.to_xml()
      parent.append(child)
    # Otherwise, we'll create the child node ourselves. We handle several possible types of value:
    #   1. If value is a dict, create an XML subtree using the dict key-value pairs.
    #   2. If the value is iterable but not a string (so a list, numpy array, tuple, set, etc.), create a
    #      comma separated list of the values and set the node's text to that.
    #   3. If the value is anything else (assumes can be cast to a reasonable string), just set the node
    #      text to that value.
    else:
      child = ET.SubElement(parent, tag)
      if isinstance(value, dict):
        for tag, value in value.items():
          self._add_subelement(child, tag, value)
      else:
        child.text = value

# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  XML utilities for HERON, independent of RAVEN.
  Replaces HERON's uses of ravenframework.utils.xmlUtils.
"""
import xml.etree.ElementTree as ET


def new_node(tag, text='', attrib=None):
  """
    Creates a new node with the given tag, text, and attributes.
    Text and attribute values are converted to strings. Unlike RAVEN's xmlUtils.newNode, illegal
    characters in tags and text are not replaced; HERON only builds nodes from known-good names.
    @ In, tag, str, the name of the node
    @ In, text, object, optional, the text of the node
    @ In, attrib, dict, optional, attribute: value pairs
    @ Out, node, xml.etree.ElementTree.Element, new node
  """
  if attrib is None:
    attrib = {}
  node = ET.Element(tag, attrib={str(key): str(value) for key, value in attrib.items()})
  node.text = str(text)
  return node

def prettify(tree):
  """
    Formats an XML tree in RAVEN's preferred style: two-space indentation, and a blank line between
    the children of the root node. Adds whitespace to the text and tails of the nodes in place.
    Matches the output of RAVEN's xmlUtils.prettify with its default arguments.
    @ In, tree, xml.etree.ElementTree.ElementTree or xml.etree.ElementTree.Element, tree to format
    @ Out, pretty, str, formatted XML
  """
  root = tree.getroot() if isinstance(tree, ET.ElementTree) else tree
  _prettify_node(root)
  # ASCII encoding (the ET.tostring default) writes non-ASCII characters as character references
  return ET.tostring(root).decode()

def _prettify_node(node, tabs=0):
  """
    Adds whitespace to a node and its children so the node is human-readable
    @ In, node, xml.etree.ElementTree.Element, node to format
    @ In, tabs, int, optional, indentation level of this node
    @ Out, None
  """
  space = '  ' * tabs
  newline_and_tab = '\n' + space
  child = None
  if node.text is None:
    node.text = ''
  if len(node):
    node.text = node.text.strip() + newline_and_tab + '  '
    for child in node:
      _prettify_node(child, tabs + 1)
    # the last child closes the node, so it gets one less indent
    child.tail = child.tail[:-2]
  node.tail = '' if node.tail is None else node.tail.strip()
  # blank lines between the children of the root node, but not after comments or processing
  # instructions, whose tags are their factory functions
  if tabs == 1 and not callable(node.tag):
    node.tail += '\n\n' + space
  else:
    node.tail += '\n' + space
  # no blank line after the last child of the root node
  if tabs == 0 and child is not None:
    child.tail = child.tail.replace('\n\n', '\n')

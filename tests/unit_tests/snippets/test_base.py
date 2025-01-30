"""
Unit tests for the base RavenSnippet class
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)
from HERON.templates.snippets import RavenSnippet
from HERON.tests.unit_tests.snippets.utils import is_subtree_matching
sys.path.pop()

import unittest
import xml.etree.ElementTree as ET


class TestRavenSnippet(unittest.TestCase):
  """ Tests for the RavenSnippet class """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.snippet = RavenSnippet()

  def test_name(self):
    """
    Test name attribute
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.snippet.name)
    name = "some_name"
    self.snippet.name = name
    self.assertEqual(self.snippet.name, name)
    self.assertEqual(self.snippet.get("name"), name)

  def test_repr(self):
    """
    Test __repr__
    @ In, None
    @ Out, None
    """
    name = "some_name"
    self.snippet.name = name
    self.assertEqual(str(self.snippet), name)

  def test_add_subelements(self):
    """
    Test add_subelements method
    @ In, None
    @ Out, None
    """
    subs = {
      "child1": "val1",
      "child2": {
        "grand1": "val2"
      }
    }
    self.snippet.add_subelements(subs)
    self.assertTrue(is_subtree_matching(self.snippet, subs))

  def test_to_assembler_node(self):
    """
    Test creating assembler node from snippet
    @ In, None
    @ Out, None
    """
    self.snippet.name = "the_name"
    self.snippet.snippet_class = "the_class"
    self.snippet.tag = "the_tag"
    assemb_tag = "assemb_tag"

    assemb = self.snippet.to_assembler_node(assemb_tag)
    self.assertEqual(assemb.tag, assemb_tag)
    self.assertEqual(assemb.get("class"), "the_class")
    self.assertEqual(assemb.get("type"), "the_tag")
    self.assertEqual(assemb.text, "the_name")

    # Only "class" and "type" attribs should be present
    assemb.attrib.pop("class")
    assemb.attrib.pop("type")
    self.assertDictEqual(assemb.attrib, {})

  def test_from_xml(self):
    """
    Test creating snippet object from XML tree
    @ In, None
    @ Out, None
    """
    xml = """
    <SomeSnippet name="some_name">
      <child1>val1</child1>
      <child2>
        <grand1>val2</grand1>
      </child2>
    </SomeSnippet>
    """
    root = ET.fromstring(xml)
    snippet_xml = RavenSnippet.from_xml(root)

    subs = {
      "child1": "val1",
      "child2": {
        "grand1": "val2"
      }
    }
    self.assertTrue(is_subtree_matching(snippet_xml, subs))

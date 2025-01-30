"""
Unit tests for the snippet factory
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.templates.snippets.factory import SnippetFactory
sys.path.pop()

import unittest
import xml.etree.ElementTree as ET


class MockBase:
  """ Base class for mock snippets """
  tag = None
  subtype = None

  @classmethod
  def get(cls, key, default=None):
    """
    Mock for xml.etree.ElementTree.Element `get` method
    @ In, key, Any, the attribute key
    @ In, default, Any, optional, default value if key not present
    @ Out, value, Any, the attribute value
    """
    return getattr(cls, key, default)


class Mock(MockBase):
  """ Mock class with no set subtype """
  tag = "mock"
  subtype = None


class MockA(MockBase):
  """ Mock class with set subtype """
  tag = "mock"
  subtype = "a"


class MockB(MockBase):
  """ Mock class with set subtype """
  tag = "mock"
  subtype = "b"


class TestSnippetFactory(unittest.TestCase):
  """ Tests for the SnippetFactory class """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.factory = SnippetFactory()

  def test_register_snippet_class(self):
    """
    Test class registration
    @ In, None
    @ Out, None
    """
    # Add a snippet class
    self.factory.register_snippet_class(Mock)
    self.assertIn(f"mock", self.factory.registered_classes)
    num_registered = len(self.factory.registered_classes)

    # Giving a new class with the same tag/subtype combo should throw an error
    class Mock2(MockBase):
      tag = "mock"
      subtype = None

    with self.assertRaises(ValueError):
      self.factory.register_snippet_class(Mock2)

    # Giving the snippet what looks like a RavenSnippet base class shouldn't raise an
    # error but it also shouldn't be added as a registered class since it has no tag.
    self.factory.register_snippet_class(MockBase)
    self.assertEqual(len(self.factory.registered_classes), num_registered)

  def test_register_all_subclasses(self):
    """
    Test register all subclasses of a class
    @ In, None
    @ Out, None
    """
    self.factory.register_all_subclasses(MockBase)
    gold = {
      "mock": Mock,
      "mock[@subType='a']": MockA,
      "mock[@subType='b']": MockB,
    }
    self.assertDictEqual(self.factory.registered_classes, gold)

  def test_has_registered_class(self):
    """
    test has_registered_class method
    @ In, None
    @ Out, None
    """
    # Add a snippet class
    self.factory.register_snippet_class(Mock)
    # See if the class is found for a matching node
    node = ET.Element("mock")
    self.assertTrue(self.factory.has_registered_class(node))
    node_a = ET.Element("mock", subType="a")
    self.assertFalse(self.factory.has_registered_class(node_a))  # not registered

  def test_get_snippet_class_key(self):
    """
    Test _get_snippet_class_key method
    @ In, None
    @ Out, None
    """
    # Get the key of a snippet-like class
    key = self.factory._get_snippet_class_key(Mock)
    self.assertEqual(key, "mock")

    # Get the key of a snippet-like class with a subtype attribute
    key = self.factory._get_snippet_class_key(MockA)
    self.assertEqual(key, "mock[@subType='a']")

  def test_get_node_key(self):
    """
    Test _get_node_key method
    @ In, None
    @ Out, None
    """
    # Get the key of an ET.Element
    node = ET.Element("element")
    key = self.factory._get_node_key(node)
    self.assertEqual(key, "element")

    # Get the key of an ET.Element with a subType attribute
    node = ET.Element("element", subType="st")
    key = self.factory._get_node_key(node)
    self.assertEqual(key, "element[@subType='st']")

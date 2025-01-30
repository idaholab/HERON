"""
Unit tests for the DataObject RavenSnippet classes
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.templates.snippets import PointSet, HistorySet, DataSet
sys.path.pop()

import unittest
import xml.etree.ElementTree as ET


class TestDataObjectBase:
  """
  Tests for the DataObject RavenSnippet base class. These are tests inherited by
  the concrete database classes and not run directly.
  """
  def test_snippet_class(self):
    """
    Test snippet_class class attribute
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.obj.snippet_class, "DataObjects")

  def test_inputs(self):
    """
    Test inputs property
    @ In, None
    @ Out, None
    """
    self.assertListEqual(self.obj.inputs, [])
    self.obj.inputs.append("inp1")
    self.assertListEqual(self.obj.inputs, ["inp1"])
    self.assertListEqual(self.obj.find("Input").text, ["inp1"])

  def test_outputs(self):
    """
    Test outputs property
    @ In, None
    @ Out, None
    """
    self.assertListEqual(self.obj.outputs, [])
    self.obj.outputs.append("out1")
    self.assertListEqual(self.obj.outputs, ["out1"])
    self.assertListEqual(self.obj.find("Output").text, ["out1"])


class TestPointSet(unittest.TestCase, TestDataObjectBase):
  """ PointSet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.obj = PointSet()

  def test_tag(self):
    """
    Test tag string
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.obj.tag, "PointSet")


class TestHistorySet(unittest.TestCase, TestDataObjectBase):
  """ HistorySet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.obj = HistorySet()

  def test_tag(self):
    """
    Test tag string
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.obj.tag, "HistorySet")


class TestDataSet(unittest.TestCase, TestDataObjectBase):
  """ DataSet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.obj = DataSet()

  def test_tag(self):
    """
    Test tag string
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.obj.tag, "DataSet")

  def test_add_index(self):
    """
    Test add index to DataSet
    @ In, None
    @ Out, None
    """
    self.obj.add_index("index_var", "some_var")
    index_node = self.obj.find("Index[@var='index_var']")
    self.assertIsNotNone(index_node)
    self.assertIn("some_var", index_node.text)

    self.obj.add_index("another_index", ["var1", "var2"])
    index_node = self.obj.find("Index[@var='another_index']")
    self.assertIsNotNone(index_node)
    self.assertListEqual(index_node.text, ["var1", "var2"])

    # Try adding duplicate index with new variables. Shouldn't add a new index node but should
    # add the values to the list.
    index_ct = len(self.obj.findall("Index"))
    self.obj.add_index("another_index", ["var3"])
    self.obj.add_index("another_index", "var4")
    self.assertEqual(len(self.obj.findall("Index")), index_ct)
    self.assertListEqual(index_node.text, ["var1", "var2", "var3", "var4"])

"""
Unit tests for the VariableGroup snippets
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.insert(0, HERON_LOC)
print(sys.path)

from HERON.templates.snippets import VariableGroup
sys.path.pop()

import unittest
import xml.etree.ElementTree as ET


class TestVariableGroup(unittest.TestCase):
  """ VariableGroup snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    # __init__ constructor
    self.group = VariableGroup(name="GRO_group")

    # from_xml constructor
    xml = "<Group name='GRO_group'>var1, var2,var3</Group>"  # NOTE: intentionally bad spacing
    root = ET.fromstring(xml)
    self.group_xml = VariableGroup.from_xml(root)

  def test_from_xml(self):
    """
    Test instantiate from XML
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.group_xml.snippet_class, "VariableGroups")
    self.assertEqual(self.group_xml.tag, "Group")
    self.assertEqual(self.group_xml.name, "GRO_group")
    self.assertListEqual(self.group_xml.variables, ["var1", "var2", "var3"])
    self.assertListEqual(self.group_xml.text, ["var1", "var2", "var3"])

  def test_default(self):
    """
    Test default values
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.group.snippet_class, "VariableGroups")
    self.assertEqual(self.group.tag, "Group")
    self.assertEqual(self.group.name, "GRO_group")
    self.assertListEqual(self.group.variables, [])
    self.assertIsNone(self.group.text)

  def test_set_variables(self):
    """
    Test variables property
    @ In, None
    @ Out, None
    """
    self.group.variables = ["var1", "var2"]
    self.assertListEqual(self.group.text, ["var1", "var2"])

    self.group.variables.append("var3")
    self.assertListEqual(self.group.text, ["var1", "var2", "var3"])

    self.group.variables.extend(["var4"])
    self.assertListEqual(self.group.text, ["var1", "var2", "var3", "var4"])

    self.group.variables.insert(0, "var0")
    self.assertListEqual(self.group.text, ["var0", "var1", "var2", "var3", "var4"])

    self.group.variables.clear()
    self.assertListEqual(self.group.text, [])

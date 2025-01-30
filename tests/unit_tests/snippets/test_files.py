"""
Unit tests for the Files snippets
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.templates.snippets import File
sys.path.pop()

import unittest
import xml.etree.ElementTree as ET


class TestFile(unittest.TestCase):
  """ Tests for File snippets """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.file = File()

    xml = "<Input name='raven_inner' type='raven'>path/to/inner.xml</Input>"
    root = ET.fromstring(xml)
    self.file_xml = File.from_xml(root)

  def test_snippet_class(self):
    """
    Test snippet_class value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.file.snippet_class, "Files")

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.file.tag, "Input")

  def test_from_xml(self):
    """
    Test instantiate from XML
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.file_xml.name, "raven_inner")
    self.assertEqual(self.file_xml.get("name"), "raven_inner")
    self.assertEqual(self.file_xml.type, "raven")
    self.assertEqual(self.file_xml.get("type"), "raven")
    self.assertEqual(self.file_xml.path, "path/to/inner.xml")
    self.assertEqual(self.file_xml.text, "path/to/inner.xml")

  def test_type(self):
    """
    Test type attribute
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.file.type)
    val = "some_type"
    self.file.type = val
    self.assertEqual(self.file.type, val)
    self.assertEqual(self.file.get("type"), val)

  def test_path(self):
    """
    Test path property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.file.path)
    path = "path/to/file.xml"
    self.file.path = path
    self.assertEqual(self.file.path, path)
    self.assertEqual(self.file.text, path)

  def test_to_assembler_node(self):
    """
    Test to_assmebler node method
    @ In, None
    @ Out, None
    """
    xml1 = """<Input name="inner_workflow" type="raven">../inner.xml</Input>"""
    file1 = File.from_xml(ET.fromstring(xml1))
    assemb1 = file1.to_assembler_node("File")
    self.assertEqual(assemb1.tag, "File")
    self.assertDictEqual(assemb1.attrib, {"class": "Files", "type": "raven"})
    self.assertEqual(assemb1.text, "inner_workflow")

    xml2 = """<Input name="heron_lib">../heron.lib</Input>"""
    file2 = File.from_xml(ET.fromstring(xml2))
    assemb2 = file2.to_assembler_node("File")
    self.assertEqual(assemb2.tag, "File")
    # getting this "type" value right is a deviation from the usual RavenSnippet.to_assemb_node() method
    self.assertDictEqual(assemb2.attrib, {"class": "Files", "type": ""})
    self.assertEqual(assemb2.text, "heron_lib")

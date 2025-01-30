"""
Unit tests for the listproperty decorator

@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""

import sys
import os
import unittest
import xml.etree.ElementTree as ET

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*3))
sys.path.append(HERON_LOC)

from HERON.templates.decorators import listproperty
sys.path.pop()


class MockListPropertyUser:
  """ A minimal class implementing an attribute as a listproperty """
  def __init__(self) -> None:
    """
    Constructor
    @ In, None
    @ Out, None
    """
    self._list = []

  @listproperty
  def prop(self) -> list[str]:
    """
    Property getter for private list variable _list
    @ In, None
    @ Out, _list, list[str], the wrapped list
    """
    return self._list

  @prop.setter
  def prop(self, lst: list[str]) -> None:
    """
    Property setter for private list variable _list
    @ In, lst, list[str], the list to set
    @ Out, None
    """
    self._list = lst


class MockListPropertyETUser:
  """ A minimal class wrapping an ET.Element object's text attribute with a listproperty """
  def __init__(self):
    """
    Constructor
    @ In, None
    @ Out, None
    """
    self.node = ET.Element("test_element")

  @listproperty
  def prop(self) -> list[str]:
    """
    Property getter for XML node text which stores a list
    @ In, None
    @ Out, _list, list[str], the wrapped list
    """
    return getattr(self.node, "text", []) or []

  @prop.setter
  def prop(self, value: list[str]) -> None:
    """
    Property setter for XML node text which stores a list
    @ In, value, list[str], the list to set
    @ Out, None
    """
    self.node.text = value


class TestListProperty(unittest.TestCase):
  def setUp(self):
    """
    Test setup
    @ In, None
    @ Out, None
    """
    self.tester = MockListPropertyUser()

  def test_get(self):
    """
    Tests property getter
    @ In, None
    @ Out, None
    """
    self.assertListEqual(self.tester.prop, [])

  def test_set(self):
    """
    Tests property setter
    @ In, None
    @ Out, None
    """
    lst = ["1", "2", "3"]
    self.tester.prop = lst
    self.assertListEqual(self.tester.prop, lst)

  def test_append(self):
    """
    Test append to list
    @ In, None
    @ Out, None
    """
    self.tester.prop.append("a")
    self.assertListEqual(self.tester.prop, ["a"])
    self.tester.prop.append("b")
    self.assertListEqual(self.tester.prop, ["a", "b"])

  def test_clear(self):
    """
    Test list clear
    @ In, None
    @ Out, None
    """
    self.tester.prop = ["b", "c"]
    self.tester.prop.clear()
    self.assertListEqual(self.tester.prop, [])

  def test_copy(self):
    """
    Test list copy
    @ In, None
    @ Out, None
    """
    lst = ["b", "c"]
    self.tester.prop = lst
    copied = self.tester.prop.copy()
    self.assertListEqual(self.tester.prop, lst)
    self.assertListEqual(copied, lst)

  def test_count(self):
    """
    Test list count
    @ In, None
    @ Out, None
    """
    lst = ["a", "b", "c", "a"]
    self.tester.prop = lst
    self.assertEqual(self.tester.prop.count("a"), lst.count("a"))
    self.assertEqual(self.tester.prop.count("b"), lst.count("b"))

  def test_extend(self):
    """
    Test list extend
    @ In, None
    @ Out, None
    """
    self.tester.prop.extend(["a", "b"])
    self.assertListEqual(self.tester.prop, ["a", "b"])
    self.tester.prop.extend(["c"])
    self.assertListEqual(self.tester.prop, ["a", "b", "c"])

  def test_index(self):
    """
    Test list index
    @ In, None
    @ Out, None
    """
    lst = ["a", "b", "c", "a"]
    self.tester.prop = lst
    self.assertEqual(self.tester.prop.index("a"), lst.index("a"))
    self.assertEqual(self.tester.prop.index("b"), lst.index("b"))
    self.assertEqual(self.tester.prop.index("c"), lst.index("c"))

  def test_insert(self):
    """
    Test insert value to list
    @ In, None
    @ Out, None
    """
    self.tester.prop = ["b", "c"]
    self.tester.prop.insert(0, "a")
    self.assertListEqual(self.tester.prop, ["a", "b", "c"])

  def test_pop(self):
    """
    Test pop value from list
    @ In, None
    @ Out, None
    """
    self.tester.prop = ["a", "b", "c"]
    popped = self.tester.prop.pop()
    self.assertEqual(popped, "c")
    self.assertListEqual(self.tester.prop, ["a", "b"])
    popped = self.tester.prop.pop(0)
    self.assertEqual(popped, "a")
    self.assertListEqual(self.tester.prop, ["b"])

    # pop from empty list
    self.tester.prop = []
    with self.assertRaises(IndexError):
      self.tester.prop.pop()

  def test_remove(self):
    """
    Test remove value from list
    @ In, None
    @ Out, None
    """
    self.tester.prop = ["b", "c"]
    self.tester.prop.remove("b")
    self.assertListEqual(self.tester.prop, ["c"])

  def test_reverse(self):
    """
    Test reverse list
    @ In, None
    @ Out, None
    """
    self.tester.prop = ["b", "c"]
    self.tester.prop.reverse()
    self.assertListEqual(self.tester.prop, ["c", "b"])

  def test_sort(self):
    """
    Test sort list
    @ In, None
    @ Out, None
    """
    lst = ["a", "b", "c", "a"]
    self.tester.prop = lst
    lst_sorted = sorted(lst)
    self.tester.prop.sort()
    self.assertListEqual(self.tester.prop, lst_sorted)


class TestListPropertyXML(unittest.TestCase):
  """
  Unit tests for the listproperty decorator wrapping an ET.Element object. This closely mimics how the listproperty
  decorator is used in practice in HERON.templates. The focus here is to test how the listproperty behaves when the
  ET.Element text is set directly with a string, then is accessed and modified through the listproperty.
  """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.tester = MockListPropertyETUser()

  def test_get_unset(self):
    """
    Test getter on node with unset text attribute
    @ In, None
    @ Out, None
    """
    self.assertIsInstance(self.tester.prop, list)
    self.assertListEqual(self.tester.prop, [])

  def test_set_with_string(self):
    """
    Test setting listproperty text with a string. String should get wrapped in a list.
    @ In, None
    @ Out, None
    """
    self.tester.node.text = "text without commas"
    self.assertIsInstance(self.tester.prop, list)
    self.assertListEqual(self.tester.prop, ["text without commas"])

  def test_set_with_string_list(self):
    """
    Test setting listproperty text with a string of comma-separated values with variable whitespace. Should
    be split into a list of values with whitespace stripped.
    @ In, None
    @ Out, None
    """
    self.tester.node.text = "val1, val2,val3,    val4  "  # intentionally ugly spacing and extra whitespace
    self.assertIsInstance(self.tester.prop, list)
    self.assertListEqual(self.tester.prop, ["val1", "val2", "val3", "val4"])

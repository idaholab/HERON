"""
Unit tests for the OutStreams snippets
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.templates.snippets import PrintOutStream, OptPathPlot, HeronDispatchPlot, TealCashFlowPlot
from HERON.tests.unit_tests.snippets.mock_classes import MockSnippet
sys.path.pop()

import unittest


class TestOutStreamBase:
  """ OutStream snippet tests """
  def test_snippet_class(self):
    """
    Test snippet class attribute
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.outstream.snippet_class, "OutStreams")

  def test_source(self):
    """
    Test source property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.outstream.find("source"))
    self.assertIsNone(self.outstream.source)

    source = "source_str"
    self.outstream.source = source
    self.assertEqual(self.outstream.source, source)
    self.assertEqual(self.outstream.find("source").text, source)

    source = MockSnippet("source_obj")
    self.outstream.source = source
    # self.assertEqual(self.outstream.source, source)
    self.assertEqual(self.outstream.source, str(source))
    self.assertEqual(self.outstream.find("source").text, source)


class TestPrintOutStream(unittest.TestCase, TestOutStreamBase):
  """ Print snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.outstream = PrintOutStream()

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.outstream.tag, "Print")

  def test_type(self):
    """
    Test "type" node text
    @ In, None
    @ Out, None
    """
    node = self.outstream.find("type")
    self.assertIsNotNone(node)
    self.assertEqual(node.text, "csv")


class TestOptPathPlot(unittest.TestCase, TestOutStreamBase):
  """ OptPathPlot snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.outstream = OptPathPlot()

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.outstream.tag, "Plot")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.outstream.subtype, "OptPath")

  def test_variables(self):
    """
    Test variables property
    @ In, None
    @ Out, None
    """
    self.assertListEqual(self.outstream.variables, [])
    self.outstream.variables.append("var1")
    self.assertListEqual(self.outstream.variables, ["var1"])
    self.assertListEqual(self.outstream.find("vars").text, ["var1"])


class TestHeronDispatchPlot(unittest.TestCase, TestOutStreamBase):
  """ HeronDispatchPlot snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.outstream = HeronDispatchPlot()

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.outstream.tag, "Plot")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.outstream.subtype, "HERON.DispatchPlot")

  def test_macro_variable(self):
    """
    Test macro_variable property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.outstream.find("macro_variable"))
    self.assertIsNone(self.outstream.macro_variable)
    macro_variable = "macro_var"
    self.outstream.macro_variable = macro_variable
    self.assertEqual(self.outstream.macro_variable, macro_variable)
    self.assertEqual(self.outstream.find("macro_variable").text, macro_variable)

  def test_micro_variable(self):
    """
    Test micro_variable property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.outstream.find("micro_variable"))
    self.assertIsNone(self.outstream.micro_variable)
    micro_variable = "micro_var"
    self.outstream.micro_variable = micro_variable
    self.assertEqual(self.outstream.micro_variable, micro_variable)
    self.assertEqual(self.outstream.find("micro_variable").text, micro_variable)

  def test_signals(self):
    """
    Test signals property
    @ In, None
    @ Out, None
    """
    self.assertListEqual(self.outstream.signals, [])
    self.outstream.signals.append("inp1")
    self.assertListEqual(self.outstream.signals, ["inp1"])
    self.assertListEqual(self.outstream.find("signals").text, ["inp1"])


class TestTealCashFlowPlot(unittest.TestCase, TestOutStreamBase):
  """ TealCashFlowPlot snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.outstream = TealCashFlowPlot()

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.outstream.tag, "Plot")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.outstream.subtype, "TEAL.CashFlowPlot")

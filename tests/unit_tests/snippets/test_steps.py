"""
Unit tests for the Steps snippets
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.templates.snippets import IOStep, MultiRun, PostProcess
from HERON.tests.unit_tests.snippets.mock_classes import MockSnippet
sys.path.pop()

import unittest


class TestStepBase:
  """ Step base class tests """
  def test_snippet_class(self):
    """
    Test snippet_class value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.step.snippet_class, "Steps")

  def test_add_item(self):
    """
    Test add_item method
    @ In, None
    @ Out, None
    """
    snippet = MockSnippet()

    assemb_tags = ["Function", "Input", "Model", "Sampler", "Optimizer", "SolutionExport", "Output"]
    add_item_funcs = {
      "Function": self.step.add_function,
      "Input": self.step.add_input,
      "Model": self.step.add_model,
      "Sampler": self.step.add_sampler,
      "Optimizer": self.step.add_optimizer,
      "SolutionExport": self.step.add_solution_export,
      "Output": self.step.add_output,
    }

    # Test allowed
    for tag in set(self.step._allowed_subs):  # set randomizes addition order
      add_item_funcs[tag](snippet)

    # Subelements should appear in order of self.assemb_tags
    tags = [sub.tag for sub in self.step]
    sorted_tags = sorted(tags, key=lambda tag: assemb_tags.index(tag))
    self.assertListEqual(tags, sorted_tags)

    # Adding unallowed tags should throw a ValueError
    for unallowed in sorted(list(set(assemb_tags) - set(self.step._allowed_subs))):
      with self.assertRaises(ValueError):
        add_item_funcs[unallowed](snippet)


class TestMultiRun(unittest.TestCase, TestStepBase):
  """ MultiRun step tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.step = MultiRun("multirun")

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.step.tag, "MultiRun")


class TestIOStep(unittest.TestCase, TestStepBase):
  """ IOStep step tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.step = IOStep("io_step")

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.step.tag, "IOStep")


class TestPostProcess(unittest.TestCase, TestStepBase):
  """ PostProcess step tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.step = PostProcess("pp")

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.step.tag, "PostProcess")

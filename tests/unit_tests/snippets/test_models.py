"""
Unit tests for the models snippets
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.templates.snippets import (RavenCode,
                                      GaussianProcessRegressor,
                                      EnsembleModel,
                                      EconomicRatioPostProcessor,
                                      ExternalModel,
                                      HeronDispatchModel,
                                      PickledROM)
sys.path.pop()

import unittest


class TestModelBase:
  """ Tests for all model classes """
  def test_snippet_class(self):
    """
    Test snippet_class value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.snippet_class, "Models")


class TestRavenCode(unittest.TestCase, TestModelBase):
  """ Tests for RavenCode model snippet """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.model = RavenCode()

  def test_tag(self):
    """
    Test XMl tag
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.tag, "Code")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.subtype, "RAVEN")

  def test_executable(self):
    """
    Test executable path property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.model.find("executable"))
    self.assertIsNone(self.model.executable)
    executable = "path/to/raven_framework"
    self.model.executable = executable
    self.assertEqual(self.model.executable, executable)
    self.assertEqual(self.model.find("executable").text, executable)

  def test_python_command(self):
    """
    Test python_command property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.model.find("clargs[@type='prepend']"))
    self.assertIsNone(self.model.python_command)
    python_command = "path/to/python -args"
    self.model.python_command = python_command
    self.assertEqual(self.model.python_command, python_command)

    found_node = False
    nodes = self.model.findall(f"clargs[@type='prepend']")
    for node in nodes:
      if node.get("arg", None) == python_command:
        found_node = True
        break
    self.assertTrue(found_node)

  def test_add_alias(self):
    """
    Test add_alias method
    @ In, None
    @ Out, None
    """
    name = "alias_name"
    suffix = "suffix"
    varname = f"{name}_{suffix}"
    self.model.add_alias(name, suffix)
    alias_text = f"Samplers|MonteCarlo@name:mc_arma_dispatch|constant@name:{varname}"
    node = self.model.find(f"alias[@variable='{varname}']")
    self.assertIsNotNone(node)
    self.assertEqual(node.text, alias_text)

  def test_set_inner_data_handling(self):
    """
    Test set_inner_data_handling method
    @ In, None
    @ Out, None
    """
    self.model.set_inner_data_handling("dest_csv", "csv")
    outstreams = self.model.findall("outputExportOutStreams")
    databases = self.model.findall("outputDatabase")
    self.assertEqual(len(outstreams), 1)
    self.assertEqual(len(databases), 0)
    self.assertEqual(outstreams[0].text, "dest_csv")

    self.model.set_inner_data_handling("dest_db", "netcdf")
    outstreams = self.model.findall("outputExportOutStreams")
    databases = self.model.findall("outputDatabase")
    self.assertEqual(len(outstreams), 0)
    self.assertEqual(len(databases), 1)
    self.assertEqual(databases[0].text, "dest_db")


class TestGaussianProcessRegressor(unittest.TestCase, TestModelBase):
  """ Tests for GaussianProcessRegressor model """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.model = GaussianProcessRegressor()

  def test_tag(self):
    """
    Test snippet XML tag
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.tag, "ROM")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.subtype, "GaussianProcessRegressor")

  def test_has_default_settings(self):
    """
    Test if default settings are all applied
    @ In, None
    @ Out, None
    """
    default_settings = {
      "alpha": 1e-8,
      "n_restarts_optimizer": 5,
      "normalize_y": True,
      "kernel": "Custom",
      "custom_kernel": "(Constant*Matern)",
      "anisotropic": True,
      "multioutput": False
    }
    for name in default_settings:
      self.assertIsNotNone(self.model.find(name))

  def test_features(self):
    """
    Test features property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.model.find("Features"))
    feats = ["feat1", "feat2"]
    self.model.features.extend(feats)
    self.assertListEqual(self.model.features, feats)
    self.assertListEqual(self.model.find("Features").text, feats)

  def test_target(self):
    """
    Test target property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.model.find("Target"))
    targets = ["val1", "val2"]
    self.model.target.extend(targets)
    self.assertListEqual(self.model.target, targets)
    self.assertListEqual(self.model.find("Target").text, targets)

  def test_custom_kernel(self):
    """
    Test custom_kernel property
    @ In, None
    @ Out, None
    """
    self.model.custom_kernel = "kern"
    self.assertEqual(self.model.custom_kernel, "kern")
    self.assertEqual(self.model.find("custom_kernel").text, "kern")


class TestEnsembleModel(unittest.TestCase, TestModelBase):
  """ EnsembleModel snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.model = EnsembleModel()

  def test_tag(self):
    """
    Test snippet XML tag
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.tag, "EnsembleModel")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.subtype, "")


class TestEconomicRatioPostProcessor(unittest.TestCase, TestModelBase):
  """ EconomicRatioPostProcessor tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.model = EconomicRatioPostProcessor()

  def test_tag(self):
    """
    Test snippet XML tag
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.tag, "PostProcessor")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.subtype, "EconomicRatio")

  def test_add_statistic(self):
    """
    Test add_statistic method
    @ In, None
    @ Out, None
    """
    self.assertEqual(len(self.model.findall("expectedValue")), 0)
    self.model.add_statistic("expectedValue", "mean", "NPV")
    self.assertEqual(len(self.model.findall("expectedValue")), 1)


class TestExternalModel(unittest.TestCase, TestModelBase):
  """ ExternalModel snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.model = ExternalModel()

  def test_tag(self):
    """
    Test snippet XML tag
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.tag, "ExternalModel")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.subtype, "")

  def test_variables(self):
    """
    Test variables property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.model.find("variables"))
    self.model.variables.append("var")
    self.assertListEqual(self.model.variables, ["var"])
    self.assertListEqual(self.model.find("variables").text, ["var"])


class TestHeronDispatchModel(unittest.TestCase, TestModelBase):
  """ HeronDispatchModel snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.model = HeronDispatchModel()

  def test_tag(self):
    """
    Test snippet XML tag
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.tag, "ExternalModel")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.subtype, "HERON.DispatchManager")


class TestPickledROM(unittest.TestCase, TestModelBase):
  """ PickledROM snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.model = PickledROM()

  def test_tag(self):
    """
    Test snippet XML tag
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.tag, "ROM")

  def test_subtype(self):
    """
    Test snippet subtype
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.model.subtype, "pickledROM")

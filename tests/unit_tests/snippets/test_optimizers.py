"""
Unit tests for the optimizers snippets
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os
import unittest

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)
from HERON.templates.snippets import BayesianOptimizer, GradientDescent
from HERON.tests.unit_tests.snippets.mock_classes import MockSnippet
from HERON.tests.unit_tests.snippets.utils import is_subtree_matching
sys.path.pop()


class TestOptimizerBase:
  """ Optimizer snippet class tests """
  def test_snippet_class(self):
    self.assertEqual(self.opt.snippet_class, "Optimizers")

  def test_objective(self):
    """
    Test objective property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.opt.find("objective"))
    self.assertIsNone(self.opt.objective)
    objective = "the_objective"
    self.opt.objective = objective
    self.assertEqual(self.opt.objective, objective)
    self.assertEqual(self.opt.find("objective").text, objective)

  def test_target_evaluation(self):
    """
    Test target_evaluation property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.opt.find("TargetEvaluation"))
    self.assertIsNone(self.opt.target_evaluation)
    snippet = MockSnippet("data_obj", "DataObjects", "PointSet")
    assemb = snippet.to_assembler_node("TargetEvaluation")
    self.opt.target_evaluation = snippet
    self.assertEqual(self.opt.target_evaluation, snippet.name)
    self.assertDictEqual(self.opt.find("TargetEvaluation").attrib, assemb.attrib)
    self.assertEqual(self.opt.find("TargetEvaluation").text, assemb.text)

  def test_base_set_opt_settings(self):
    """
    Test set_opt_settings method
    @ In, None
    @ Out, None
    """
    # no settings; test if no error is thrown
    self.opt.set_opt_settings({})

    # some settings
    settings = {
      "convergence": {"some_param": "some_val"},
      "persistence": 2,
      "limit": 10,
      "type": "min"
    }
    self.opt.set_opt_settings(settings)

    self.assertEqual(self.opt.find(f"convergence/some_param").text, settings["convergence"]["some_param"])
    self.assertEqual(self.opt.find("convergence/persistence").text, settings["persistence"])
    self.assertEqual(self.opt.find("samplerInit/limit").text, settings["limit"])
    self.assertEqual(self.opt.find("samplerInit/type").text, settings["type"])


class TestBayesianOptimizer(unittest.TestCase, TestOptimizerBase):
  """ BayesianOptimizer snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.opt = BayesianOptimizer()

  def test_has_default_settings(self):
    """
    Test default settings
    @ In, None
    @ Out, None
    """
    default_settings = {
      "samplerInit": {
        "limit": 100,
        "type": "max",
        "writeSteps": "every",
      },
      "ModelSelection": {
        "Duration": 1,
        "Method": "Internal"
      },
      "convergence": {
        "acquisition": 1e-5,
        "persistence": 4
      },
      "Acquisition": ""
    }
    self.assertTrue(is_subtree_matching(self.opt, default_settings))

  def test_set_opt_settings(self):
    """
    Test set_opt_settings method
    @ In, None
    @ Out, None
    """
    # no settings; test if no error is thrown
    self.opt.set_opt_settings({})
    self.assertIsNone(self.opt.find("samplerInit/initialSeed"))  # not set without passing in with settings dict
    self.assertIsNotNone(self.opt.find(f"Acquisition/ExpectedImprovement"))  # defaults to EI

    # acquisition functions
    for acquisition_func in ["ExpectedImprovement", "ProbabilityOfImprovement", "LowerConfidenceBound"]:
      settings = {"algorithm": {"BayesianOpt": {"acquisition": acquisition_func}}}
      self.opt.set_opt_settings(settings)
      self.assertEqual(len(self.opt.find(f"Acquisition")), 1)  # exactly 1 acquisition function
      self.assertIsNotNone(self.opt.find(f"Acquisition/{acquisition_func}"))

    # model selection
    settings = {"algorithm": {"BayesianOpt": {
      "ModelSelection": {
        "Duration": 10,
        "Method": "other_method"
      },
    }}}
    self.opt.set_opt_settings(settings)
    self.assertEqual(self.opt.find("ModelSelection/Duration").text, 10)
    self.assertEqual(self.opt.find("ModelSelection/Method").text, "other_method")

    # seed
    settings = {"algorithm": {"BayesianOpt": {"seed": 12345}}}
    self.opt.set_opt_settings(settings)
    self.assertEqual(self.opt.find("samplerInit/initialSeed").text, 12345)

  def test_set_sampler(self):
    """
    Test set_sampler method
    @ In, None
    @ Out, None
    """
    sampler = MockSnippet("sampler", "Samplers", "Stratified")
    self.opt.set_sampler(sampler)
    node = self.opt.find("Sampler")
    self.assertDictEqual(node.attrib, sampler.to_assembler_node("Sampler").attrib)
    self.assertEqual(node.text, sampler.name)

  def test_set_rom(self):
    """
    Test set_rom method
    @ In, None
    @ Out, None
    """
    rom = MockSnippet("gpr", "Models", "ROM")
    self.opt.set_rom(rom)
    node = self.opt.find("ROM")
    self.assertDictEqual(node.attrib, rom.to_assembler_node("ROM").attrib)
    self.assertEqual(node.text, rom.name)


class TestGradientDescent(unittest.TestCase, TestOptimizerBase):
  """ GradientDescent snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.opt = GradientDescent()

  def test_has_default_settings(self):
    """
    Test default settings
    @ In, None
    @ Out, None
    """
    default_settings = {
      "samplerInit": {
        "limit": 800,
        "type": "max",
        "writeSteps": "every"
      },
      "gradient": {  # CentralDifference, SPSA not exposed in HERON input
        "FiniteDifference": ""  # gradDistanceScalar option not exposed
      },
      "convergence": {
        "persistence": 1,
        "gradient": 1e-4,
        "objective": 1e-8
      },
      "stepSize": {
        "GradientHistory": {
          "growthFactor": 2,
          "shrinkFactor": 1.5,
          "initialStepScale": 0.2
        }
      },
      "acceptance": {
        "Strict": ""
      }
    }
    self.assertTrue(is_subtree_matching(self.opt, default_settings))

  def test_set_opt_settings(self):
    """
    Test set_opt_settings method
    @ In, None
    @ Out, None
    """
    # no settings; test if no error is thrown
    self.opt.set_opt_settings({})

    # gradient history settings
    settings = {"algorithm" : {"GradientDescent": {
      "growthFactor": 10,
      "shrinkFactor": 11,
      "initialStepScale": 12
    }}}
    self.opt.set_opt_settings(settings)
    self.assertEqual(self.opt.find("stepSize/GradientHistory/growthFactor").text, 10)
    self.assertEqual(self.opt.find("stepSize/GradientHistory/shrinkFactor").text, 11)
    self.assertEqual(self.opt.find("stepSize/GradientHistory/initialStepScale").text, 12)

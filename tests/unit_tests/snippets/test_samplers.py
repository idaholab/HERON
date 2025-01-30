"""
Unit tests for the Samplers snippets
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os
import unittest
import xml.etree.ElementTree as ET

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)
from HERON.templates.snippets import RavenSnippet, SampledVariable, Sampler, Grid, MonteCarlo, Stratified, CustomSampler, EnsembleForward
from HERON.tests.unit_tests.snippets.mock_classes import MockSnippet
# from HERON.tests.unit_tests.snippets.utils import is_subtree_matching
sys.path.pop()


class TestSampledVariable(unittest.TestCase):
  """ Tests for the SampledVariable class """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.var = SampledVariable()

  def test_snippet_class(self):
    """
    Test snippet_class value
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.var.snippet_class)

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.var.tag, "variable")

  def test_initial(self):
    """
    Test initial property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.var.find("initial"))
    self.var.initial = 12345
    self.assertEqual(self.var.find("initial").text, 12345)

  def test_distribution(self):
    """
    Test distribution property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.var.find("distribution"))
    dist = MockSnippet("my_dist", "Distributions", "Uniform")
    self.var.distribution = dist
    self.assertEqual(self.var.find("distribution").text, "my_dist")


class TestSamplerBase(unittest.TestCase):
  """ Tests for Sampler snippets """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.sampler = Sampler()

  def test_snippet_class(self):
    """
    Test snippet_class value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.sampler.snippet_class, "Samplers")

  def test_num_sampled_vars(self):
    """
    Test num_sampled_vars attribute
    @ In, None
    @ Out, None
    """
    num_vars = len(self.sampler.findall("variable"))
    ET.SubElement(self.sampler, "variable")
    new_num_vars = len(self.sampler.findall("variable"))
    self.assertEqual(new_num_vars - num_vars, 1)
    self.assertEqual(new_num_vars, self.sampler.num_sampled_vars)

  def test_denoises(self):
    """
    Test denoises property
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.sampler.find("constant[@name='denoises']"))
    self.sampler.denoises = 10
    self.assertEqual(self.sampler.find("constant[@name='denoises']").text, 10)

  def test_init_seed(self):
    """
    Test init_seed property
    @ In, None
    @ Out, None
    """
    self.sampler.init_seed = 10
    self.assertEqual(self.sampler.find("samplerInit/initialSeed").text, 10)

  def test_init_limit(self):
    """
    Test init_limit property
    @ In, None
    @ Out, None
    """
    self.sampler.init_limit = 10
    self.assertEqual(self.sampler.find("samplerInit/limit").text, 10)

  def test_add_variable(self):
    """
    Test add_variable method
    @ In, None
    @ Out, None
    """
    mock_var = ET.Element("variable", name="mock_var")
    self.sampler.add_variable(mock_var)
    self.assertIsNotNone(self.sampler.find("variable[@name='mock_var']"))

  def test_add_constant(self):
    """
    Test add_constant method
    @ In, None
    @ Out, None
    """
    self.sampler.add_constant("my_const", "some_value")
    self.assertEqual(self.sampler.find("constant[@name='my_const']").text, "some_value")

  def test_has_variable(self):
    """
    Test has_variable method
    @ In, None
    @ Out, None
    """
    sampled_var = SampledVariable("some_var")
    self.sampler.add_variable(sampled_var)
    self.assertTrue(self.sampler.has_variable(sampled_var))
    self.assertTrue(self.sampler.has_variable(sampled_var.name))

    other_var = SampledVariable("other_var")  # not added to sampler
    self.assertFalse(self.sampler.has_variable(other_var))
    self.assertFalse(self.sampler.has_variable(other_var.name))

class TestGrid(unittest.TestCase):
  """ Grid sampler snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.sampler = Grid()

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.sampler.tag, "Grid")


class TestMonteCarlo(unittest.TestCase):
  """ MonteCarlo sampler snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.sampler = MonteCarlo()

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.sampler.tag, "MonteCarlo")


class TestStratified(unittest.TestCase):
  """ Stratified sampler snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.sampler = Stratified()

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.sampler.tag, "Stratified")

  def test_has_samplerinit(self):
    """
    Test if sampler has <samplerInit> node
    @ In, None
    @ Out, None
    """
    self.assertIsNotNone(self.sampler.find("samplerInit"))


class TestCustomSampler(unittest.TestCase):
  """ CustomSampler sampler snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.sampler = CustomSampler()

  def test_tag(self):
    """
    Test tag value
    @ In, None
    @ Out, None
    """
    self.assertEqual(self.sampler.tag, "CustomSampler")


class TestEnsembleForward(unittest.TestCase):
  """ EnsembleForward sampler snippet tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.sampler = EnsembleForward()

    xml = """
    <EnsembleForward name="ensemble_sampler">
      <Grid name="grid"/>
      <MonteCarlo name="mc"/>
      <Stratified name="lhs"/>
      <CustomSampler name="custom"/>
      <NotARegisteredSampler name="some_sampler"/>
    </EnsembleForward>
    """
    self.sampler_xml = EnsembleForward.from_xml(ET.fromstring(xml))

  def test_from_xml(self):
    """
    Test instantiate from XML
    @ In, None
    @ Out, None
    """
    self.assertIsInstance(self.sampler_xml.find("Grid"), Grid)
    self.assertIsInstance(self.sampler_xml.find("MonteCarlo"), MonteCarlo)
    self.assertIsInstance(self.sampler_xml.find("Stratified"), Stratified)
    self.assertIsInstance(self.sampler_xml.find("CustomSampler"), CustomSampler)
    self.assertNotIsInstance(self.sampler_xml.find("NotARegisteredSampler"), RavenSnippet)

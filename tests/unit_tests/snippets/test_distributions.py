"""
Unit tests for the Distributions RavenSnippet classes
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import sys
import os

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.templates.snippets import distributions
sys.path.pop()

import unittest
import xml.etree.ElementTree as ET

# Limited testing of all allowable distributions
# We can easily test the construction (with __init__ or from_xml) by looping over distribution names.
# It's more difficult to automate testing of property getting/setting, so we reserve that resolution of
# testing for a couple of the most commonly used distributions.

dists = [
  "Beta",
  "Exponential",
  "Gamma",
  "Laplace",
  "Logistic",
  "LogNormal",
  "LogUniform",
  "Normal",
  "Triangular",
  "Uniform",
  "Weibull",
  "Custom1D",
  "Bernoulli",
  "Binomial",
  "Geometric",
  "Poisson",
  "Categorical",
  "UniformDiscrete",
  "MarkovCategorical",
  "MultivariateNormal",
  "NDInverseWeight",
  "NDCartesianSpline"
]

class TestDistributions(unittest.TestCase):
  """ Tests for all distribution classes """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.dist_classes = {dist_name: getattr(distributions, dist_name) for dist_name in dists}

  def test_constructed_distributions(self):
    """
    Test distribution object construction
    @ In, None
    @ Out, None
    """
    for dist_name, cls in self.dist_classes.items():
      self.assertEqual(cls.snippet_class, "Distributions")
      self.assertEqual(cls.tag, dist_name)

  def test_from_xml(self):
    """
    Test instantiate from XML
    @ In, None
    @ Out, None
    """
    for dist_name, cls in self.dist_classes.items():
      xml = f"<{dist_name} name='new_dist'/>"
      root = ET.fromstring(xml)
      dist = cls.from_xml(root)

      self.assertEqual(dist.snippet_class, "Distributions")
      self.assertEqual(dist.tag, dist_name)
      self.assertEqual(dist.name, "new_dist")

# Explicit testing of just a couple of the most common distributions we expect to see in HERON.

class TestUniform(unittest.TestCase):
  """ Tests for the Uniform distribution class """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.dist = distributions.Uniform()

  def test_bounds(self):
    """
    Test distribution bounds properties
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.dist.lower_bound)
    self.assertIsNone(self.dist.upper_bound)

    self.dist.lower_bound = 0
    self.dist.upper_bound = 1

    self.assertEqual(self.dist.lower_bound, 0)
    self.assertEqual(self.dist.upper_bound, 1)
    self.assertEqual(self.dist.find("lowerBound").text, 0)
    self.assertEqual(self.dist.find("upperBound").text, 1)


class TestNormal(unittest.TestCase):
  """ Normal distribution tests """
  def setUp(self):
    """
    Tester setup
    @ In, None
    @ Out, None
    """
    self.dist = distributions.Normal()

  def test_params(self):
    """
    Test mean and sigma properties
    @ In, None
    @ Out, None
    """
    self.assertIsNone(self.dist.mean)
    self.assertIsNone(self.dist.sigma)

    self.dist.mean = 0
    self.dist.sigma = 1

    self.assertEqual(self.dist.mean, 0)
    self.assertEqual(self.dist.sigma, 1)
    self.assertEqual(self.dist.find("mean").text, 0)
    self.assertEqual(self.dist.find("sigma").text, 1)

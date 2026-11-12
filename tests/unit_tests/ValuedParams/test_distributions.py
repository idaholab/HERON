# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Tests that the distributions HERON supports in <uncertainty> (src/ValuedParams/distributions.py)
  are supported by RAVEN, which samples them. This test requires RAVEN.
"""
import os
import sys
import math
import unittest
import xml.etree.ElementTree as ET

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)
from HERON.src.ValuedParams import distributions
from HERON.src.ValuedParams.RandomVariable import RandomVariable
sys.path.pop()

from ravenframework import Distributions as RavenDistributions
from ravenframework import MessageHandler

# Example input for each supported distribution, as a user would write it in <uncertainty>
EXAMPLES = {
  'Uniform': '<Uniform name="d"><lowerBound>9</lowerBound><upperBound>11</upperBound></Uniform>',
  'Normal': '<Normal name="d" verbosity="quiet"><mean>10</mean><sigma>0.05</sigma></Normal>',
  'LogNormal': '<LogNormal name="d"><mean>0.3</mean><sigma>0.7</sigma><low>2</low></LogNormal>',
  'Triangular': '<Triangular name="d"><min>9</min><max>11</max><apex>10</apex></Triangular>',
  'Beta': '<Beta name="d"><low>9</low><high>11</high><alpha>2</alpha><beta>3</beta></Beta>',
  'Gamma': '<Gamma name="d"><low>9</low><alpha>2.5</alpha><beta>1.5</beta></Gamma>',
  'Exponential': '<Exponential name="d"><lambda>1.5</lambda><low>9</low></Exponential>',
  'Weibull': '<Weibull name="d"><k>1.7</k><lambda>1.3</lambda><low>9</low></Weibull>',
  'LogUniform': '<LogUniform name="d"><lowerBound>1</lowerBound><upperBound>2</upperBound><base>natural</base></LogUniform>',
  'Logistic': '<Logistic name="d"><location>10</location><scale>0.5</scale></Logistic>',
  'Laplace': '<Laplace name="d"><location>10</location><scale>0.5</scale></Laplace>',
  'Custom1D': ('<Custom1D name="d"><workingDir>{data_dir}</workingDir><functionType>cdf</functionType>'
               '<dataFilename>custom1d.csv</dataFilename><functionID>cdf</functionID><variableID>x</variableID></Custom1D>'),
  'Poisson': '<Poisson name="d"><mu>3</mu></Poisson>',
  'Binomial': '<Binomial name="d"><n>10</n><p>0.4</p></Binomial>',
  'Bernoulli': '<Bernoulli name="d"><p>0.4</p></Bernoulli>',
  'Geometric': '<Geometric name="d"><p>0.4</p></Geometric>',
  'UniformDiscrete': ('<UniformDiscrete name="d"><lowerBound>9</lowerBound><upperBound>11</upperBound>'
                      '<nPoints>3</nPoints><strategy>withReplacement</strategy></UniformDiscrete>'),
  'Categorical': '<Categorical name="d"><state outcome="9">0.5</state><state outcome="11">0.5</state></Categorical>',
  'MarkovCategorical': ('<MarkovCategorical name="d"><state outcome="9" index="1"/><state outcome="11" index="2"/>'
                        '<transition>-1 1 1 -1</transition></MarkovCategorical>'),
}

def _type_name(content):
  """
    Gets a comparable description of an InputData content or attribute type.
    @ In, content, InputTypes.InputType or None, the type
    @ Out, name, tuple or None, the type's name and, for enumerated types, its allowed values
  """
  if content is None:
    return None
  return (content.getName(), tuple(getattr(content, 'enumList', None) or ()))

class TestDistributionsSupportedByRaven(unittest.TestCase):
  """
    Each distribution HERON supports must be a one-dimensional RAVEN distribution that RAVEN can read
    exactly as HERON writes it into the RAVEN workflow.
  """
  @classmethod
  def setUpClass(cls):
    """
      Collects the RAVEN distribution specs and a quiet RAVEN message handler.
      @ In, None
      @ Out, None
    """
    cls.raven_specs = {spec.getName(): spec for spec in RavenDistributions.returnInputParameter().subs}
    cls.message_handler = MessageHandler.MessageHandler()
    cls.message_handler.initialize({'verbosity': 'silent'})
    cls.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'distribution_data')

  def test_examples_cover_supported(self):
    """
      Every supported distribution has an example, so adding a distribution requires adding its example here.
      @ In, None
      @ Out, None
    """
    self.assertEqual(set(EXAMPLES), {dist.name for dist in distributions.SUPPORTED})

  def test_structure_matches_raven(self):
    """
      Every node and attribute HERON accepts is accepted by RAVEN with the same type, so RAVEN can read
      any input HERON accepts. RAVEN may accept more than HERON.
      @ In, None
      @ Out, None
    """
    for dist in distributions.SUPPORTED:
      with self.subTest(distribution=dist.name):
        self.assertIn(dist.name, self.raven_specs, 'not a RAVEN distribution')
        self._compare(distributions.make_spec(dist), self.raven_specs[dist.name])

  def _compare(self, heron, raven):
    """
      Checks that one HERON spec node is accepted by the matching RAVEN spec node, recursively.
      @ In, heron, InputData.ParameterInput, HERON spec node
      @ In, raven, InputData.ParameterInput, RAVEN spec node with the same name
      @ Out, None
    """
    path = heron.getName()
    self.assertEqual(_type_name(heron.contentType), _type_name(raven.contentType), f'<{path}> content type')
    # ordering only matters for nodes with subnodes
    if raven.subs:
      self.assertEqual(heron.subOrder is None, raven.subOrder is None, f'<{path}> ordering')
    for attr, info in heron.parameters.items():
      self.assertIn(attr, raven.parameters, f'<{path}> attribute "{attr}"')
      self.assertEqual(_type_name(info['type']), _type_name(raven.parameters[attr]['type']), f'<{path}> "{attr}" type')
      self.assertEqual(info['required'], raven.parameters[attr]['required'], f'<{path}> "{attr}" required')
    raven_subs = {sub.getName(): sub for sub in raven.subs}
    for sub in heron.subs:
      self.assertIn(sub.getName(), raven_subs, f'<{path}> subnode <{sub.getName()}>')
      self._compare(sub, raven_subs[sub.getName()])

  def test_raven_reads_heron_output(self):
    """
      RAVEN reads the distribution XML that HERON writes into the workflow, and it is one-dimensional
      with a usable inverse CDF.
      @ In, None
      @ Out, None
    """
    for name, example in EXAMPLES.items():
      with self.subTest(distribution=name):
        raven_dist = self._read_with_raven(self._heron_output(example.format(data_dir=self.data_dir)))
        # not every RAVEN distribution sets its dimensionality, so check its class instead
        self.assertNotIsInstance(raven_dist, RavenDistributions.NDimensionalDistributions)
        self.assertTrue(math.isfinite(float(raven_dist.ppf(0.5))))

  def _heron_output(self, example):
    """
      Gets the distribution XML that HERON writes into the RAVEN workflow for an <uncertainty> input.
      @ In, example, str, distribution XML as a user would write it
      @ Out, node, xml.etree.ElementTree.Element, distribution XML as HERON writes it
    """
    spec = RandomVariable.get_input_specs()()
    spec.parseNode(ET.fromstring(f'<uncertainty>{example}</uncertainty>'))
    vp = RandomVariable()
    vp.read('test_component', spec, 'sweep')
    return vp.get_distribution()

  def _read_with_raven(self, node):
    """
      Reads a distribution node the way RAVEN reads the <Distributions> block of a workflow.
      @ In, node, xml.etree.ElementTree.Element, distribution XML
      @ Out, dist, ravenframework.Distributions.Distribution, the initialized RAVEN distribution
    """
    block = ET.Element('Distributions')
    block.append(node)
    collection = RavenDistributions.returnInputParameter()
    collection.parseNode(block)
    parsed = collection.subparts[0]
    dist = RavenDistributions.factory.returnInstance(parsed.getName())
    dist.messageHandler = self.message_handler
    dist.handleInput(parsed, globalAttributes={})
    return dist

if __name__ == '__main__':
  unittest.main()

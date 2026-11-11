# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Tests reading the <uncertainty> ValuedParam (RandomVariable) and its user manual entry.
"""
import os
import sys
import unittest
import xml.etree.ElementTree as ET

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)
from HERON.src import Cases, Components, messages
from HERON.src.ValuedParams import distributions
from HERON.src.ValuedParams.RandomVariable import RandomVariable, MANUAL_LABEL
sys.path.pop()

class RecordingBackend:
  """
    Message backend that records errors instead of printing them, following messages.MessageBackend.
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.errors = [] # (error type, message) for each error raised

  def message(self, caller, message, tag, verbosity, color=None, forcePrint=False):
    """
      Ignores a message.
      @ In, caller, object, object sending the message
      @ In, message, str, the message
      @ In, tag, str, message kind, e.g. 'Warning'
      @ In, verbosity, str, verbosity level of the message
      @ In, color, str, optional, unused
      @ In, forcePrint, bool, optional, unused
      @ Out, None
    """

  def error(self, caller, etype, message, tag='ERROR', verbosity='silent', color=None):
    """
      Records an error and raises it.
      @ In, caller, object, object raising the error
      @ In, etype, type, exception type to raise
      @ In, message, str, the error message
      @ In, tag, str, optional, unused
      @ In, verbosity, str, optional, unused
      @ In, color, str, optional, unused
      @ Out, None
    """
    self.errors.append((etype, message))
    raise etype(message)

def read(xml):
  """
    Reads an <uncertainty> node the way HERON reads it from an input file.
    @ In, xml, str, the <uncertainty> node
    @ Out, vp, RandomVariable, the ValuedParam that was read
  """
  spec = RandomVariable.get_input_specs()()
  spec.parseNode(ET.fromstring(xml))
  vp = RandomVariable()
  vp.read('test_component', spec, 'sweep')
  return vp

class TestRandomVariableRead(unittest.TestCase):
  """
    Reading an <uncertainty> node.
  """
  def setUp(self):
    """
      Sends HERON messages to a recording backend.
      @ In, None
      @ Out, None
    """
    self.backend = RecordingBackend()
    messages.set_default_backend(self.backend)

  def tearDown(self):
    """
      Restores the default message backend.
      @ In, None
      @ Out, None
    """
    messages.set_default_backend(None)

  def test_supported_distribution(self):
    """
      A supported distribution is passed on to the RAVEN workflow, with values converted to their types.
      @ In, None
      @ Out, None
    """
    vp = read('<uncertainty><Normal name="price_dist"><mean>10000</mean><sigma>2</sigma></Normal></uncertainty>')
    dist = vp.get_distribution()
    self.assertEqual(dist.tag, 'Normal')
    self.assertEqual(dist.attrib, {'name': 'price_dist'})
    self.assertEqual([(sub.tag, sub.text) for sub in dist], [('mean', '10000.0'), ('sigma', '2.0')])

  def test_nested_distribution(self):
    """
      Attributes and text of a distribution's subnodes are passed on to the RAVEN workflow.
      @ In, None
      @ Out, None
    """
    vp = read('<uncertainty><Categorical name="d"><state outcome="9">0.25</state><state outcome="low">0.75</state>'
              '</Categorical></uncertainty>')
    states = vp.get_distribution().findall('state')
    self.assertEqual([(state.attrib, state.text) for state in states],
                     [({'outcome': 9.0}, '0.25'), ({'outcome': 'low'}, '0.75')])

  def test_multidimensional_rejected(self):
    """
      Multi-dimensional RAVEN distributions are not accepted.
      @ In, None
      @ Out, None
    """
    for name in ['MultivariateNormal', 'NDInverseWeight', 'NDCartesianSpline', 'NDimensionalDistributions']:
      with self.subTest(distribution=name):
        with self.assertRaisesRegex(IOError, f'Unrecognized input node "{name}"'):
          read(f'<uncertainty><{name} name="d"/></uncertainty>')

  def test_misspelled_rejected(self):
    """
      A misspelled distribution, or a misspelled node in a supported distribution, is not accepted.
      @ In, None
      @ Out, None
    """
    with self.assertRaisesRegex(IOError, 'Unrecognized input node "Norml"'):
      read('<uncertainty><Norml name="d"><mean>1</mean></Norml></uncertainty>')
    with self.assertRaisesRegex(IOError, 'Unrecognized input node "meen"'):
      read('<uncertainty><Normal name="d"><meen>1</meen><sigma>1</sigma></Normal></uncertainty>')

  def test_missing_distribution(self):
    """
      An <uncertainty> without a distribution raises a HERON error.
      @ In, None
      @ Out, None
    """
    with self.assertRaises(RuntimeError):
      read('<uncertainty/>')
    self.assertEqual(len(self.backend.errors), 1)
    self.assertIn('no distribution found', self.backend.errors[0][1])

def find_specs(spec, name, seen=None):
  """
    Finds every subnode spec with the given name in a spec tree.
    @ In, spec, InputData.ParameterInput, the root of the spec tree
    @ In, name, str, node name to find
    @ In, seen, set, optional, specs already searched
    @ Out, found, list(InputData.ParameterInput), specs named name
  """
  if seen is None:
    seen = set()
  found = []
  for sub in spec.subs:
    if sub.getName() == name:
      found.append(sub)
    elif sub not in seen:
      seen.add(sub)
      found.extend(find_specs(sub, name, seen))
  return found

class TestRandomVariableManual(unittest.TestCase):
  """
    The user manual documents the distributions once, and every <uncertainty> refers to that section.
  """
  @classmethod
  def setUpClass(cls):
    """
      Finds the <uncertainty> specs in the Case and Component specs.
      @ In, None
      @ Out, None
    """
    cls.uncertainty_specs = []
    for top in (Cases.Case, Components.Component):
      cls.uncertainty_specs.extend(find_specs(top.get_input_specs(), 'uncertainty'))

  def test_found_uncertainty(self):
    """
      The search finds the places <uncertainty> is allowed (cash flow parameters among them).
      @ In, None
      @ Out, None
    """
    self.assertGreaterEqual(len(self.uncertainty_specs), 4)

  def test_reference_not_expansion(self):
    """
      Each <uncertainty> entry refers to the uncertainty section instead of listing the distributions,
      whether its manual entry is generated from the spec class or an instance of it.
      @ In, None
      @ Out, None
    """
    for spec in self.uncertainty_specs:
      for source in (spec, spec()):
        with self.subTest(source=type(source).__name__):
          tex = source.generateLatex(recDepth=2)
          self.assertIn('\\item \\xmlNode{uncertainty}', tex)
          self.assertIn(f'\\ref{{{MANUAL_LABEL}}}', tex)
          for dist in distributions.SUPPORTED:
            self.assertNotIn(f'\\xmlNode{{{dist.name}}}', tex)

  def test_distributions_documented(self):
    """
      Each supported distribution has its own manual entry.
      @ In, None
      @ Out, None
    """
    for dist in distributions.SUPPORTED:
      with self.subTest(distribution=dist.name):
        self.assertIn(f'\\item \\xmlNode{{{dist.name}}}', distributions.make_spec(dist).generateLatex(recDepth=1))

if __name__ == '__main__':
  unittest.main()

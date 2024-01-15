
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the ValuedParam entity.
  These are objects that need to return values, but come from
  a wide variety of different sources.
"""
from .ValuedParam import ValuedParam, InputData, InputTypes
from ravenframework.Distributions import returnInputParameter
from ravenframework.utils import xmlUtils

CF_TARGET_MAP = {
  'reference_price': 'alpha',
  'driver': 'driver',
  'scaling_factor_x': 'scale',
  'reference_driver': 'reference',
}

# class for potentially dynamically-evaluated quantities
class RandomVariable(ValuedParam):
  """
    Represents a ValuedParam that takes values directly from synthetic histories
    sampled in RAVEN.
  """
  # these types represent values that do not need to be evaluated at run time, as they are determined.
  @classmethod
  def get_input_specs(cls):
    """
      Template for parameters that can take a scalar, an ARMA history, or a function
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('UQ', contentType=InputTypes.StringType,
        descr=r"""indicates that this value will be taken from synthetically-generated signals,
              which will be provided to the dispatch at run time by RAVEN from trained models. The value
              of this node should be the name of a synthetic history generator in the
              \xmlNode{DataGenerator} node.""")
    # grabbing DistributionsCollection which has all Distribution specs as subnodes
    dist_collection = returnInputParameter()
    for sub in dist_collection.subs:
      spec.addSub(sub)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._source_kind = 'UQ'  # name of ValuedParam
    self._distribution = None # instance of provided RAVEN distribution

  def read(self, comp_name, spec, mode, alias_dict=None):
    """
      Used to read valued param from XML input
      @ In, comp_name, str, name of component that this valued param will be attached to; only used for print messages
      @ In, spec, InputData params, input specifications
      @ In, mode, type of simulation calculation
      @ In, alias_dict, dict, optional, aliases to use for variable naming
      @ Out, needs, list, signals needed to evaluate this ValuedParam at runtime
    """
    super().read(comp_name, spec, mode, alias_dict=None)
    # aliases get used to convert variable names, notably for the cashflow's "capacity"
    if alias_dict is None:
      alias_dict = {}
    sub_distributions = spec.subparts
    if len(sub_distributions) == 0:
      self.raiseAnError(RuntimeError, 'No distribution found.')
    else:
      sub_distribution = sub_distributions[0]
    self._distribution = self.convert_spec_to_xml(sub_distribution)
    return []

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, value, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, meta, dict, dictionary of meta (possibly changed during evaluation)
    """
    if aliases is None:
      aliases = {}

    # oh boy, this is wild
    try:
      dist_name = self._distribution.attrib['name']
      key = dist_name.split('_dist')[0]
    except:
      self.raiseAnError(RuntimeError, 'RandomVariable VP distribution name not set properly.')

    if CF_TARGET_MAP[target_var] not in key:
      self.raiseAnError(RuntimeError, 'Target variable not matching with RandomVariable VP distribution.')

    value = inputs['HERON']['RAVEN_vars'][key][0]
    return {target_var: value}, inputs


### HELPERS
  def get_distribution(self):
    """
      Returns distribution XML node from user.
      @ In, None
      @ Out, distribution, XML node
    """
    return self._distribution

  def convert_spec_to_xml(self, input_spec, rec_depth=0):
    """
      Returns distribution XML node from user.
      @ In, input, InputData, value-based spec
      @ In, recDepth, int, how many levels deep are we in recursive calls
      @ Out, distribution, XML node
    """
    xml = xmlUtils.newNode(input_spec.name)
    if input_spec.parameterValues:
      xml.attrib = input.parameterValues
    if input_spec.value != '':
      xml.text = str(input_spec.value)
    if input_spec.subparts:
      for sub in input_spec.subparts:
        if 'ravenframework.utils.InputData' in str(type(sub)):
          sub_node = self.convert_spec_to_xml(sub, rec_depth+1)
          xml.append(sub_node)
    return xml

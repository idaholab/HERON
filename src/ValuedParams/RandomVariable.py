
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

# TODO: this is temporary until it becomes a part of the ValuedParam registry
CF_TARGET_MAP = {
  'reference_price': 'alpha',
  'driver': 'driver',
  'scaling_factor_x': 'scale',
  'reference_driver': 'reference',
}

# class for potentially dynamically-evaluated quantities
class RandomVariable(ValuedParam):
  """
    Represents a ValuedParam that takes values directly from a sampled distribution. Users specify
    a distribution from RAVEN and values are sampled using a RAVEN MonteCarlo sampler.
  """
  # these types represent values that do not need to be evaluated at run time, as they are determined.
  @classmethod
  def get_input_specs(cls):
    """
      Template for parameters that can take a value sampled from a distribution.
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('uncertainty', contentType=InputTypes.StringType,
        descr=r"""indicates that this value is a random variable whose value is sampled from a
              distribution which needs to be provided by the user. A subnode must be provided
              which matches one of the RAVEN Distributions from `ravenframework/Distributions`.
              """)
    # grabbing DistributionsCollection which has all Distribution specs as subnodes
    dist_collection = returnInputParameter()
    # NOTE: any input errors (missing XML attribs or subnodes) will have been found when parsing XML
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
    self._source_kind = 'uncertainty'  # name of ValuedParam
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
    # get_input_specs() will have already generated all RAVEN distribution templates, now find the one the user provided
    sub_distributions = spec.subparts
    # check that the user provided *something*
    if len(sub_distributions) == 0:
      msg = 'RandomVariable ValuedParam was requested but no distribution found in HERON XML input.'
      self.raiseAnError(RuntimeError,msg)
    else:
      # seems something is there in a list, get the first instance (multiple distributions doesn't make sense)
      sub_distribution = sub_distributions[0]
    # now convert the given XML Distribution subnode from the RAVEN InputData/specs back to XML
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

    # getting the correct key to extract RAVEN variable
    try:
      # this assumes that HERON template driver overwrote the name of the distribution
      # something like `{comp_name}_{cashflow_name}_{cashflow_attrib}_dist`
      dist_name = self._distribution.attrib['name']
      key = dist_name.split('_dist')[0]
    except KeyError as e:
      msg = 'RandomVariable VP distribution name not set properly.'
      raise KeyError(msg) from e
    except RuntimeError as e:
      # just in case there is some other error
      msg = 'RandomVariable VP distribution not found.'
      raise RuntimeError(msg) from e

    # mapping from `template_driver` naming convention to HERON Dispatch Manager naming convention
    # also serves as a running list of acceptable HERON inputs that can take RandomVariable ValuedParam
    if CF_TARGET_MAP[target_var] not in key:
      msg = 'Target variable not matching with RandomVariable VP distribution.'
      self.raiseAnError(RuntimeError, msg)

    # extract raven variable (already compiled in HERON Dispatch Manager upstream of this)
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
      Returns XML node from user.
      TODO: This should be moved to RAVEN in the near future.
      @ In, input_spec, InputData, value-based spec
      @ In, rec_depth, int, how many levels deep are we in recursive calls
      @ Out, xml, XML node
    """
    xml = xmlUtils.newNode(input_spec.name)
    if input_spec.parameterValues:
      xml.attrib = input_spec.parameterValues
    if input_spec.value != '':
      xml.text = str(input_spec.value)
    if input_spec.subparts:
      for sub in input_spec.subparts:
        if 'ravenframework.utils.InputData' in str(type(sub)):
          sub_node = self.convert_spec_to_xml(sub, rec_depth+1)
          xml.append(sub_node)
    return xml

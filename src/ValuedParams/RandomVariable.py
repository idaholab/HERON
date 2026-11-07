
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the ValuedParam entity.
  These are objects that need to return values, but come from
  a wide variety of different sources.
"""
import copy

from .ValuedParam import ValuedParam
from . import distributions
from ravenframework.utils import InputData, InputTypes
from HERON.src.xml_utils import new_node

# label of the user manual section that documents <uncertainty> and its distributions
MANUAL_LABEL = 'sec:uncertainty'

_DESCR = r"""indicates that this value is a random variable whose value is sampled from a
    distribution which needs to be provided by the user. A subnode must be provided
    which matches one of the supported distributions."""

def _generate_latex_reference(recDepth=0, sectionLevel=1):
  """
    Generates the user manual entry for the <uncertainty> node as a reference to its own section.
    Replaces InputData's generateLatex on the <uncertainty> spec, so the arguments match it.
    @ In, recDepth, int, optional, recursion depth (indentation level) of this node
    @ In, sectionLevel, int, optional, unused; kept to match InputData's generateLatex
    @ Out, msg, str, LaTeX for this node
  """
  descr = f'{_DESCR} See Section~\\ref{{{MANUAL_LABEL}}} for the supported distributions.'
  return (InputData.doDent(recDepth) + '\\item \\xmlNode{uncertainty}:\n' +
          InputData.wrapText(descr, indent=InputData.doDent(recDepth, 1)) + '\n')

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
    one of the distributions HERON supports (see distributions.py), and values are sampled using a
    RAVEN MonteCarlo sampler.
  """
  # these types represent values that do not need to be evaluated at run time, as they are determined.
  @classmethod
  def get_input_specs(cls):
    """
      Template for parameters that can take a value sampled from a distribution.
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('uncertainty', contentType=InputTypes.StringType, descr=_DESCR)
    for dist in distributions.SUPPORTED:
      spec.addSub(distributions.make_spec(dist))
    # Distributions HERON does not support are deprecated but still accepted. Non-strict mode keeps
    # them as raw XML in "additionalInput" so they can be passed to RAVEN unchanged.
    spec.setStrictMode(False)
    # The supported distributions are documented once in their own manual section, so each place
    # <uncertainty> is allowed refers there instead of repeating them.
    spec.generateLatex = staticmethod(_generate_latex_reference)
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
    # supported distributions were parsed into subparts; any others were kept as raw XML in additionalInput
    if spec.subparts:
      # use the first distribution (multiple distributions doesn't make sense)
      # now convert the given XML Distribution subnode from the InputData specs back to XML
      self._distribution = self.convert_spec_to_xml(spec.subparts[0])
    elif spec.additionalInput:
      unsupported = spec.additionalInput[0]
      self.raiseAWarning(f'Distribution <{unsupported.tag}> in <uncertainty> for component "{comp_name}" is not '
                         'supported by HERON and has been deprecated; it is passed to RAVEN without being checked. '
                         f'Supported distributions: {", ".join(distributions.supported_names())}')
      # copy so that changes to the distribution (e.g. renaming) do not alter the user's input
      self._distribution = copy.deepcopy(unsupported)
    else:
      # check that the user provided *something*
      msg = 'RandomVariable ValuedParam was requested but no distribution found in HERON XML input.'
      self.raiseAnError(RuntimeError,msg)
    return []

  def evaluate(self, inputs, target_var=None, aliases=None, custom_input=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ In, custom_input, list, optional, additional input nodes from user input
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
    xml = new_node(input_spec.name)
    if input_spec.parameterValues:
      xml.attrib = input_spec.parameterValues
    if input_spec.value != '':
      xml.text = str(input_spec.value)
    if input_spec.subparts:
      for sub in input_spec.subparts:
        if isinstance(sub, InputData.ParameterInput):
          sub_node = self.convert_spec_to_xml(sub, rec_depth+1)
          xml.append(sub_node)
    return xml


# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Custom user-defined dynamically-evaluated quantities
"""
from .ValuedParam import ValuedParam, InputData, InputTypes

# class for custom dynamically-evaluated quantities
class Function(ValuedParam):
  """
    Represents a ValuedParam that takes values from a custom user-defined Python method.
  """

  @classmethod
  def get_input_specs(cls):
    """
      Template for parameters that can take a scalar, an ARMA history, or a function
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('Function', contentType=InputTypes.StringType,
        descr=r"""indicates this value should be taken from a Python function, as described
        in the \xmlNode{DataGenerators} node.""")
    spec.addParam('method', param_type=InputTypes.StringType,
        descr=r"""the name of the \xmlNode{DataGenerator} from which this value should be taken.""")
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._method_name = None # name of the method within the module
    self._source_kind = 'Function'

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
    self._source_name = spec.value
    self._method_name = spec.parameterValues['method']
    return [self._method_name]

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, data, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, meta, dict, dictionary of meta (possibly changed during evaluation)
    """
    if aliases is None:
      aliases = {}
    # TODO how to handle aliases for functions?
    # the "request" is what we're asking for from the function, the first argument given.
    # -> note it can be None if the function is not a transfer-type function
    request = inputs.pop('request', None)
    data, meta = self._target_obj.evaluate(self._method_name, request, inputs)
    return data, meta

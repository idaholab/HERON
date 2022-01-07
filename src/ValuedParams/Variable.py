
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Values taken from the RAVEN variable soup (in the inner)
"""
from .ValuedParam import ValuedParam, InputData, InputTypes

# class for custom dynamically-evaluated quantities
class Variable(ValuedParam):
  """
    Represents a ValuedParam that takes value from the RAVEN inner variable set
  """

  @classmethod
  def get_input_specs(cls):
    """
      Template for parameters that can take a scalar, an ARMA history, or a function
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('variable', contentType=InputTypes.StringType,
        descr=r"""the name of the variable from inner RAVEN variables.""")
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._raven_var = None # name of RAVEN variable

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
    self._raven_var = spec.value
    return [self._raven_var]

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
    key = self._raven_var if target_var is None else target_var
    var = aliases.get(key, self._raven_var)
    try:
      val = inputs['HERON']['RAVEN_vars'][var]
    except KeyError as e:
      msg = f'ERROR: requested variable "{var}" not found among RAVEN variables!'
      msg += '  -> Available:'
      for vn in inputs['HERON']['RAVEN_vars'].keys():
        msg += f'       {vn}'
      self.raiseAnError(RuntimeError, msg)
    return {key: float(val)}, inputs

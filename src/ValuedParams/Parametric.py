
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Values that are swept, optimized, or fixed in the "outer" workflow,
  so end up being constants in the "inner" workflow.
"""
from .ValuedParam import ValuedParam, InputData, InputTypes

# class for custom dynamically-evaluated quantities
class Parametric(ValuedParam):
  """
    Represents a ValuedParam that takes fixed values from parametrization in the outer.
  """

  @classmethod
  def get_input_specs(cls):
    """
      Template for parameters that can take a scalar, an ARMA history, or a function
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('Parametric', contentType=InputTypes.StringType,
        descr=r"""indicates this value should be parametrized (or fixed) in the outer run,
        but functionally act as a constant in the inner workflow.""")
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._parametric = None # (low, high) for opt, sweep values for sweep, or fixed value for fixed
    self._value = None      # value is uncertain in outer, but fixed in inner. Set during inner.
    # NOTE I think "value" may only be set in the DispatchManager currently, and possibly
    #  only for Capacities. Perhaps we need a registry for valued params that keeps track
    #  of them for setting purposes.

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
    self._parametric = spec.value
    return []

  def set_value(self, value):
    """
      Set the value for this parametric source.
      @ In, value, float, value
      @ Out, None
    """
    self._value = value

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, value, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, meta, dict, dictionary of meta (possibly changed during evaluation)
    """
    data = {target_var: self._value}
    return data, inputs

######
# dummy classes, just for changing descriptions, but they act the same as parameteric
class FixedValue(Parametric):
  @classmethod
  def get_input_specs(cls):
    """
      make specifications
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('fixed_value', contentType=InputTypes.FloatType,
        descr=r"""indicates this value should be fixed in the outer run,
        and act as a constant in the inner workflow.""")
    return spec

class OptBounds(Parametric):
  @classmethod
  def get_input_specs(cls):
    """
      make specifications
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('opt_bounds', contentType=InputTypes.FloatListType,
        descr=r"""indicates this value should be optimized in the outer run,
        while acting as a constant in the inner workflow.""")
    return spec

class SweepValues(Parametric):
  @classmethod
  def get_input_specs(cls):
    """
      make specifications
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('sweep_values', contentType=InputTypes.FloatListType,
        descr=r"""indicates this value should be parametrically swept in the outer run,
        while acting as a constant in the inner workflow.""")
    return spec

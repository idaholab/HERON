
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
    spec.addParam('debug_value', param_type=InputTypes.FloatType, required=False,
        descr=r"""provides a value to use for this entry when in Case mode has "debug" enabled.""")
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._parametric = None  # (low, high) for opt, sweep values for sweep, or fixed value for fixed
    self._debug_value = None # value to use if in debug mode (if desired by user)
    # NOTE that _parametric gets FIXED in the inner runs, becoming a constant
    # NOTE I think "_parametric" may only be set in the DispatchManager currently, and possibly
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
    self._debug_value = spec.parameterValues.get('debug_value', None)
    return []

  def get_value(self, debug=False):
    """
      Get the value for this parametric source.
      @ In, debug, bool, optional if True then give debug value (if any)
      @ Out, value, None, value
    """
    if debug and self._debug_value is not None:
      return self._debug_value
    else:
      return self._parametric

  def set_value(self, value):
    """
      Set the value for this parametric source.
      Usually done in the Inner to fix a sampled property
      @ In, value, float, value
      @ Out, None
    """
    self._parametric = value

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, value, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, meta, dict, dictionary of meta (possibly changed during evaluation)
    """
    data = {target_var: self._parametric}
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
    spec = super().get_input_specs()
    spec.name = 'fixed_value'
    spec.contentType = InputTypes.FloatType
    spec.description = r"""indicates this value should be fixed in the outer run,
                       and act as a constant in the inner workflow."""
    return spec

class OptBounds(Parametric):
  @classmethod
  def get_input_specs(cls):
    """
      make specifications
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = super().get_input_specs()
    spec.name = 'opt_bounds'
    spec.contentType = InputTypes.FloatListType
    spec.description = r"""indicates this value should be optimized in the outer run,
        while acting as a constant in the inner workflow."""
    return spec

class SweepValues(Parametric):
  @classmethod
  def get_input_specs(cls):
    """
      make specifications
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = super().get_input_specs()
    spec.name = 'sweep_values'
    spec.contentType = InputTypes.FloatListType
    spec.description = r"""indicates this value should be parametrically swept in the outer run,
        while acting as a constant in the inner workflow."""
    return spec

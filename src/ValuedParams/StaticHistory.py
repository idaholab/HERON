
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the ValuedParam entity StaticHistory.
  These are objects that need to return values, but come from
  a wide variety of different sources.
"""
from .ValuedParam import ValuedParam, InputData, InputTypes

# class for potentially dynamically-evaluated quantities
class StaticHistory(ValuedParam):
  """
    Represents a ValuedParam that takes values directly from a static CSV file.
  """

  @classmethod
  def get_input_specs(cls):
    """
      Template for parameters that can take a variable from a CSV file.
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory(
      'CSV',
      contentType=InputTypes.StringType,
      descr=r"""indicates that this value will be taken from a static CSV file signals,
              which will be provided to the dispatch at run time. The value
              of this node should be the name of a static history CSV generator found in the
              \xmlNode{DataGenerator} node."""
    )
    spec.addParam(
      'variable',
      param_type=InputTypes.StringType,
      descr=r"""indicates which variable coming from static history CSV this value should be taken from."""
    )
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._var_name = None # name of the variable within the static hist
    self._source_kind = 'CSV'

  def read(self, comp_name, spec, mode, alias_dict=None):
    """
      Used to read ValuedParam from XML input
      @ In, comp_name, str, name of component that this valued param will be attached to; only used for print messages
      @ In, spec, InputData params, input specifications
      @ In, mode, type of simulation calculation
      @ In, alias_dict, dict, optional, aliases to use for variable naming
      @ Out, needs, list, signals needed to evaluate this ValuedParam at runtime
    """
    super().read(comp_name, spec, mode, alias_dict=None)
    alias_dict = {} if alias_dict is None else alias_dict
    self._source_name = spec.value
    self._var_name = spec.parameterValues['variable']
    return [self._var_name]

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, value, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, meta, dict, dictionary of meta (possibly changed during evaluation)
    """
    aliases = {} if aliases is None else aliases
    # set the outgoing name for the evaluation results
    key = self._var_name if not target_var else target_var
    # allow aliasing of complex variable names
    var_name = aliases.get(self._var_name, self._var_name)
    # the year/cluster indices have already been handled by the Dispatch Manager
    # -> we just need to know the time index
    t = inputs['HERON']['time_index']
    try:
      value = inputs['HERON']['RAVEN_vars'][var_name][t]
    except KeyError:
      self.raiseAnError(
        RuntimeError,
        f'variable "{var_name}" was not found among the RAVEN variables!' +
        f'Please check to ensure the provided CSV file contains "{var_name}" as a header'
      )
    except IndexError:
      val = inputs['HERON']['RAVEN_vars'][var_name]
      self.raiseAnError(
        RuntimeError,
        f'Attempted to access variable "{var_name}" beyond the end of its length! ' +
        f'Requested index {t} but max index is {len(val)-1}'
      )
    else:
      return {key: value}, inputs


# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the ValuedParam entity.
  These are objects that need to return values, but come from
  a wide variety of different sources.
"""
from .ValuedParam import ValuedParam, InputData, InputTypes

# class for potentially dynamically-evaluated quantities
class SyntheticHistory(ValuedParam):
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
    spec = InputData.parameterInputFactory('ARMA', contentType=InputTypes.StringType,
        descr=r"""indicates that this value will be taken from synthetically-generated signals,
              which will be provided to the dispatch at run time by RAVEN from trained models. The value
              of this node should be the name of a synthetic history generator in the
              \xmlNode{DataGenerator} node.""")
    spec.addParam('variable', param_type=InputTypes.StringType,
        descr=r"""indicates which variable coming from the synthetic histories this value should be taken from.""")
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._var_name = None # name of the variable within the synth hist
    self._source_kind = 'ARMA'

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
    if aliases is None:
      aliases = {}
    # set the outgoing name for the evaluation results
    key = self._var_name if not target_var else target_var
    # allow aliasing of complex variable names
    var_name = aliases.get(self._var_name, self._var_name)
    # the year/cluster indices have already been handled by the Dispatch Manager
    # -> we just need to know the time index
    t = inputs['HERON']['time_index']
    try:
      value = inputs['HERON']['RAVEN_vars'][var_name][t]
    except KeyError as e:
      self.raiseAnError(RuntimeError, f'variable "{var_name}" was not found among the RAVEN variables!')
    except IndexError as e:
      val = inputs['HERON']['RAVEN_vars'][var_name]
      self.raiseAnError(RuntimeError, f'Attempted to access variable "{var_name}" beyond the end of its length! ' +
                        f'Requested index {t} but max index is {len(val)-1}')
    return {key: value}, inputs

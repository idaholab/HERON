
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Values that are expressed as linear ratios of one another.
  Primarily intended for transfer functions.
"""
from .ValuedParam import ValuedParam, InputData, InputTypes
from .Polynomial import Polynomial

ONE_TUP = tuple([1])

class Linear(Polynomial):
  """
    Represents a ValuedParam that is a linearized transfer function.
  """

  @classmethod
  def get_input_specs(cls):
    """
      Define parameters for a linear transfer function.
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('linear', contentType=InputTypes.StringType,
        descr=r"""indicates this value should be interpreted as a ratio based on an input value.""")
    rate = InputData.parameterInputFactory('rate', contentType=InputTypes.FloatType,
        descr=r"""linear coefficient for the indicated \xmlAttr{resource}.""")
    rate.addParam('resource', param_type=InputTypes.StringType,
        descr=r"""indicates the resource for which the linear transfer rate is being provided in this node.""")
    spec.addSub(rate)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()

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
    for rate_node in spec.findAll('rate'):
      resource = rate_node.parameterValues['resource']
      # linear coefficients are all order 1, no cross-variable multipyling
      # coefficient signs come from activity, not coeffs, so fix that up too
      self._coefficients[tuple([resource])][ONE_TUP] = abs(rate_node.value)
    return []

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, balance, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, inputs, dict, dictionary of meta (possibly changed during evaluation)
    """
    # TODO is this an unnecessarily slow check?
    if target_var not in [x[0] for x in self._coefficients]:
      self.raiseAnError(RuntimeError, f'"rate" for target variable "{target_var}" not found for ' +
                        f'ValuedParam {self.name}!')
    req_res, req_amt = next(iter(inputs['request'].items()))
    req_rate = self._coefficients[(req_res)][(1)]
    balance = {req_res: req_amt}
    for res, rate in self._coefficients.items():
      if res == req_res:
        continue
      balance[res] = rate / req_rate * req_amt
    return balance, inputs

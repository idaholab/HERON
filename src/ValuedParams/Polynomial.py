
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Values that are expressed as a polynomial relationship.
  For example, ax^2 + bxy + cy^2 + dx + ey = fm^2 + gmn + hn^2 + im + jn + k
  Primarily intended for transfer functions.
"""
from collections import defaultdict

from .ValuedParam import ValuedParam, InputData, InputTypes

class Polynomial(ValuedParam):
  """
    Represents a ValuedParam that is a polynomial relationship.
  """

  @classmethod
  def get_input_specs(cls):
    """
      Define parameters for a polynomial transfer function.
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('poly', contentType=InputTypes.StringType,
        descr=r"""indicates this transfer function is expressed by a polynomial relationship of arbitrary order.
                  Note the polynomial must be specified in weak form, with all terms on one side of the equation
                  set equal to zero. For instance, the equation $ax^2 + bx + c = dy^2 + fy + g$ should be reformulated
                  as $ax^2 + bx + (c-g) - dy^2 - fy = 0$.""")
    coeff = InputData.parameterInputFactory('coeff', contentType=InputTypes.FloatType,
        descr=r"""one coefficient for one poloynomial term of the specified \xmlAttr{resources}.""")
    coeff.addParam('resource', param_type=InputTypes.StringListType,
        descr=r"""indicates the resource(s) for which the polynomial coefficient is being provided in this node.
                  Note that the order of the resources matters for specifying the polynomial \xmlAttr{order}.""")
    coeff.addParam('order', param_type=InputTypes.FloatListType,
        descr=r"""indicates the orders of the polynomial for each resource specified, in order.
                  For example, if \xmlAttr{resources} is ``x, y'', then order ``2,3'' would mean
                  the specified coefficient is for $x^{2}y^{3}$.""")
    spec.addSub(coeff)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._coefficients = defaultdict(dict) # coeffs, stored as {(resources): {(order): coeff}}

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
    for coeff_node in spec.findAll('coeff'):
      resource = coeff_node.parameterValues['resource']
      order = coeff_node.parameterValues['order']
      # CHECKME does this preserve order correctly?
      self._coefficients[tuple(resource)][tuple(order)] = coeff_node.value
    return []

  def get_coefficients(self):
    """
      Returns linear coefficients.
      @ In, None
      @ Out, coeffs, dict, coefficient mapping
    """
    return self._coefficients

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, balance, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, inputs, dict, dictionary of meta (possibly changed during evaluation)
    """
    if target_var not in self._coefficients:
      self.raiseAnError(RuntimeError, f'"rate" for target variable "{target_var}" not found for ' +
                        f'ValuedParam {self.name}!')
    req_res, req_amt = next(iter(inputs['request'].items()))
    req_rate = self._coefficients[req_res]
    balance = {req_res: req_amt}
    for res, rate in self._coefficients.items():
      if res == req_res:
        continue
      balance[res] = rate / req_rate * req_amt
    return balance, inputs

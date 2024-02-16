
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Transfer fucntions that are expressed as a polynomial relationship.
  For example, ax^2 + bxy + cy^2 + dx + ey = fm^2 + gmn + hn^2 + im + jn + k
"""
from collections import defaultdict

from .TransferFunc import TransferFunc, InputData, InputTypes

class Polynomial(TransferFunc):
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
        descr=r"""one coefficient for one poloynomial term of the specified \xmlAttr{resources}.
                  Care should be taken to assure the sign of the coefficient is working as expected,
                  as consumed resources have a negative sign while produced resources have a positive
                  sign, and the full equation should have the form 0 = ... .""")
    coeff.addParam('resource', param_type=InputTypes.StringListType,
        descr=r"""indicates the resource(s) for which the polynomial coefficient is being provided in this node.
                  Note that the order of the resources matters for specifying the polynomial \xmlAttr{order}.""")
    coeff.addParam('order', param_type=InputTypes.IntegerListType,
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
    # coeffs, stored as { (resources): { (order): coeff, }, }
    # e.g., {(water, flour): {(1,1): 42, (1,2): 3.14},
    #        (water): {(1): 1.61} }
    self._coefficients = defaultdict(dict)

  def read(self, comp_name, spec):
    """
      Used to read valued param from XML input
      @ In, comp_name, str, name of component that this valued param will be attached to; only used for print messages
      @ In, spec, InputData params, input specifications
      @ Out, None
    """
    super().read(comp_name, spec)
    for coeff_node in spec.findFirst('poly').findAll('coeff'):
      resource = coeff_node.parameterValues['resource']
      order = coeff_node.parameterValues['order']
      self._coefficients[tuple(resource)][tuple(order)] = coeff_node.value

  def get_resources(self):
    """
      Provides the resources used in this transfer function.
      @ In, None
      @ Out, resources, set, set of resources used
    """
    res_set = set()
    for res_tup, ord_dict in self._coefficients.items():
      res_set = res_set.union(set(res_tup))
    return res_set

  def get_coefficients(self):
    """
      Returns linear coefficients.
      @ In, None
      @ Out, coeffs, dict, coefficient mapping
    """
    return self._coefficients

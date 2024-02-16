
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Values that are expressed as linear ratios of one another.
  Primarily intended for transfer functions.
"""
import numpy as np

from .TransferFunc import TransferFunc, InputData, InputTypes

# class for custom dynamically-evaluated quantities
class Ratio(TransferFunc):
  """
    Represents a TransferFunc that uses a linear balance of resources, such as 3a + 7b -> 2c.
    This means the ratios of the resources must be maintained, NOT 3a + 7b = 2c!
  """

  @classmethod
  def get_input_specs(cls):
    """
      Input specification for this class.
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('ratio', contentType=InputTypes.StringType,
        descr=r"""indicates this transfer function is a constant linear combination of resources. For example,
              a balance equation might be written as 3a + 7b -> 2c, implying that to make 2c, it always takes
              3 parts a and 7 parts b, or the balance ratio (3a, 7b, 2c). This means that the ratio of (3, 7, 2) must be
              maintained between (a, b, c) for all production levels. Note that the coefficient signs are automatically fixed
              internally to be negative for consumed quantities and positive for produced resources, regardless of signs used
              by the user. For an equation-based transfer function instead of balance ratio, see Polynomial.""")
    rate = InputData.parameterInputFactory('rate', contentType=InputTypes.FloatType,
        descr=r"""linear coefficient for the indicated \xmlAttr{resource}.""")
    rate.addParam('resource', param_type=InputTypes.StringType,
        descr=r"""indicates the resource for which the linear transfer ratio is being provided in this node.""")
    spec.addSub(rate)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._coefficients = None # ratios, stored in a dict as key: value

  def read(self, comp_name, spec):
    """
      Used to read transfer func from XML input
      @ In, comp_name, str, name of component that this valued param will be attached to; only used for print messages
      @ In, spec, InputData params, input specifications
      @ Out, None
    """
    super().read(comp_name, spec)
    self._coefficients = {}
    node = spec.findFirst('ratio')
    # ALIAS SUPPORT
    if node is None:
      node = spec.findFirst('linear')
      if node is None:
        self.raiseAnError(IOError, f'Unrecognized transfer function for component "{comp_name}": "{spec.name}"')
      self.raiseAWarning('"linear" has been deprecated and will be removed in the future; see "ratio" transfer function!')
    for rate_node in node.findAll('rate'):
      resource = rate_node.parameterValues['resource']
      self._coefficients[resource] = rate_node.value

  def get_resources(self):
    """
      Provides the resources used in this transfer function.
      @ In, None
      @ Out, resources, set, set of resources used
    """
    return set(self._coefficients.keys())

  def get_coefficients(self):
    """
      Returns linear coefficients.
      @ In, None
      @ Out, coeffs, dict, coefficient mapping
    """
    return self._coefficients

  def set_io_signs(self, consumed, produced):
    """
      Fix up input/output signs, if interpretable
      @ In, consumed, list, list of resources consumed in the transfer
      @ In, produced, list, list of resources produced in the transfer
      @ Out, None
    """
    for res, coef in self.get_coefficients().items():
      if res in consumed:
        self._coefficients[res] = - np.abs(coef)
      elif res in produced:
        self._coefficients[res] = np.abs(coef)
      else:
        # should not be able to get here after IO gets checked!
        raise RuntimeError('While checking transfer coefficient, resource "{res}" was neither consumed nor produced!')


# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the TransferFunc entity.
  These define the transfer functions for generating Components.
"""
import sys
from HERON.src import _utils as hutils
try:
  import ravenframework
except ModuleNotFoundError:
  framework_path = hutils.get_raven_loc()
  sys.path.append(framework_path)
from ravenframework.utils import InputData, InputTypes
from ravenframework.BaseClasses import MessageUser

# class for potentially dynamically-evaluated quantities
class TransferFunc(MessageUser):
  """
    These define the transfer functions for generating Components.
  """

  @classmethod
  def get_input_specs(cls, name):
    """
      Define inputs for this VP.
      @ In, name, string, name for spec (tag)
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory(name)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.type = self.__class__.__name__ # class type, for easy checking

  def __repr__(self) -> str:
    """
      Return Object Representation String
      @ In, None
      @ Out, None
    """
    return "<HERON TransferFunc>"

  def read(self, comp_name, spec):
    """
      Used to read transfer function from XML input
      @ In, comp_name, str, name of component that this transfer function is describing
      @ In, spec, InputData params, input specifications
      @ Out, needs, list, signals needed to evaluate this ValuedParam at runtime
    """

  def check_io(self, inputs, outputs, comp_name):
    """
      Checks that the transfer function uses all and only the resources used by the component.
      @ In, inputs, list, list of input resources to check against.
      @ In, outputs, list, list of output resources to check against.
      @ In, comp_name, str, name of component that this transfer function is describing
      @ Out, None
    """
    used = self.get_resources()
    inps = set(inputs)
    outs = set(outputs)
    excess_inputs = inps - used
    excess_outputs = outs - used
    unrecog = used - inps.union(outs)
    if excess_inputs or excess_outputs or unrecog:
      msg = f'Transfer function for Component "{comp_name}" has a mismatch with consumed and produced!'
      msg += f'\n... All Consumed: {inps}'
      msg += f'\n... All Produced: {outs}'
      msg += f'\n... All in Transfer Function: {used}'
      if excess_inputs:
        msg += f'\n... Consumed but not used in transfer: {excess_inputs}'
      if excess_outputs:
        msg += f'\n... Produced but not used in transfer: {excess_outputs}'
      if unrecog:
        msg += f'\n... In transfer but not consumed or produced: {unrecog}'
      self.raiseAnError(IOError, msg)

  def set_io_signs(self, consumed, produced):
    """
      Fix up input/output signs, if interpretable
      @ In, consumed, list, list of resources consumed in the transfer
      @ In, produced, list, list of resources produced in the transfer
      @ Out, None
    """
    # nothing to do by default

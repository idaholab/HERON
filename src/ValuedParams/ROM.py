
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the ValuedParam entity.
  These are objects that need to return values, but come from
  a wide variety of different sources.
"""
import os
import sys
import numpy as np

from HERON.src._utils import get_raven_loc
from .ValuedParam import ValuedParam, InputData, InputTypes


class ROM(ValuedParam):
  """
    Represents a ValuedParam that takes values from RAVEN ReducedOrderModel instances.
    The evaluation involves sending in expected inputs and receiving outputs.
  """
  # these types represent values that do not need to be evaluated at run time, as they are determined.
  @classmethod
  def get_input_specs(cls):
    """
      Template for parameters that can take a scalar, an ARMA history, or a function
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    # this VP needs sub-VPs for sources of inputs to the ROM
    from .Factory import factory as vp_factory
    spec = InputData.parameterInputFactory('ROM', contentType=InputTypes.StringType,
        descr=r"""indicates that this value will be taken from a RAVEN-trained ROM.""")
    # which ROM is it?
    spec.addParam('rom', param_type=InputTypes.StringType, required=True,
        descr=r"""indicates which DataGenerator ROM should be used for this value.""")
    # which variable(s?) to take from ROM
    spec.addParam('variable', param_type=InputTypes.StringType, required=True,
        descr=r"""indicates the variable output of the ROM from which this value should be taken.""")
    # inputs to the ROM
    descr = r"""designates the source of one of the inputs to the ROM."""
    allowed = ['fixed_value', 'sweep_values', 'opt_bounds', 'variable', 'ARMA', 'Function']
    inp = vp_factory.make_input_specs('input', descr=descr, allowed=allowed)
    descr = r"""name of the variable to pass into the ROM evaluation. Determined by the ROM training."""
    inp.addParam('name', required=True, descr=descr)
    spec.addSub(inp)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._source_kind = 'ROM'
    self._source_name = None   # name of source ROM
    self._inputs = {}          # map of {name: VP} for all input sources to ROM
    self._output = None        # name of output that should be used

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
    # which data generator ROM?
    self._source_name = spec.parameterValues['rom']
    # which ROM output to use for value?
    self._output = spec.parameterValues['variable']
    # use VPs for inputs to the ROM
    req_signals = [self._source_name]
    for vp_node in spec.findAll('input'):
      name = vp_node.parameterValues['name']
      new_signals = self.make_sub_vp(name, comp_name, vp_node, mode)
      req_signals.extend(new_signals)
    return req_signals

  def make_sub_vp(self, name, comp, spec, mode):
    """
      Creates a sub VP for this VP
      @ In, name, str, input variable name for ROM
      @ In, comp, str, name of associated component
      @ In, spec, InputData params, input specifications
      @ In, mode, type of simulation calculation
      @ Out, signals, list, variable names needed for evaluation
    """
    from ValuedParamHandler import ValuedParamHandler
    vp = ValuedParamHandler(name)
    signal = vp.read(comp, spec, mode)
    self._inputs[name] = {'vp': vp, 'signals': [signal]}
    return signal

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, run information from RAVEN, including meta and other run info
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, value, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, inputs, dict, possibly-modified dictionary of run information
    """
    if aliases is None:
      aliases = {}
    # set the outgoing name for the evaluation results
    key = self._var_name if not target_var else target_var
    # allow aliasing of complex variable names
    var_name = aliases.get(self._output, self._output)
    # the year/cluster indices have already been handled by the Dispatch Manager
    rlz = {}
    ## get the inputs from the input VPs
    for inp_name, inp_info in self._inputs.items():
      vp = inp_info['vp']
      res, _ = vp.evaluate(inputs, target_var=inp_name, aliases=aliases)
      rlz[inp_name] = np.atleast_1d(res[inp_name])
    ## run ROM
    res = self._target_obj.evaluate(rlz)[var_name]
    # NOTE assuming always at least 1d, which is true for RAVEN ROMs I think?
    # TODO can we check this using something more robust?
    if len(res) > 1:
      t = inputs['HERON']['time_index']
      res = res[t]
    else:
      res = res[0]
    return {key: res}, inputs

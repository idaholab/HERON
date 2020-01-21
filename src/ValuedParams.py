"""
  Defines the Component entity.
"""
from __future__ import unicode_literals, print_function
import os
import sys

raven_path = '~/projects/raven/framework' # TODO plugin RAVEN path
sys.path.append(os.path.expanduser(raven_path))
from utils import InputData,InputTypes

# class for potentially dynamically-evaluated quantities
class ValuedParam:
  """
    This class enables the identification of runtime-evaluated variables
    with a variety of sources (fixed values, parametric values, data histories, function evaluations, etc).
  """
  # these types represent values that do not need to be evaluated at run time, as they are determined.
  valued_methods = ['fixed_value', 'sweep_values', 'opt_bounds']

  @classmethod
  def get_input_specs(cls, name):
    """
      Template for parameters that can take a scalar, an ARMA history, or a function
      @ In, name, string, name for spec (tag)
      @ In, skip, list, optional, elements to not include
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory(name)
    # for when the value is fixed (takes precedence over "sweep" and "opt") ...
    spec.addSub(InputData.parameterInputFactory('fixed_value', contentType=InputTypes.FloatType))
    # for when the value is parametric (only in "sweep" mode)
    spec.addSub(InputData.parameterInputFactory('sweep_values', contentType=InputTypes.FloatListType))
    # for when the value is optimized (only in "min" or "max" mode)
    spec.addSub(InputData.parameterInputFactory('opt_bounds', contentType=InputTypes.FloatListType))
    # for when the value is time-dependent and given by an ARMA
    arma = InputData.parameterInputFactory('ARMA', contentType=InputTypes.StringType)
    arma.addParam('variable', param_type=InputTypes.StringType)
    spec.addSub(arma)
    # for when the value comes from evaluating a function
    func = InputData.parameterInputFactory('Function', contentType=InputTypes.StringType)
    func.addParam('method', param_type=InputTypes.StringType)
    spec.addSub(func)
    # for when the value comes from another variable
    var = InputData.parameterInputFactory('variable', contentType=InputTypes.StringType)
    spec.addSub(var)
    # for when the result obtained needs to grow from year to year
    growth = InputData.parameterInputFactory('growth', contentType=InputTypes.FloatType)
    growth_mode = InputTypes.makeEnumType('growthType', 'growthType', ['linear', 'exponential'])
    growth.addParam('mode', param_type=growth_mode)
    spec.addSub(growth)
    return spec

  def __init__(self, name):
    """
      Constructor.
      @ In, name, str, name of this valued param
      @ Out, None
    """
    self.name = name         # member whom this ValuedParam provides values, e.g. Component.economics.alpha
    self.type = None         # source (e.g. ARMA, Function, variable)
    self._comp = None        # component who uses this valued param
    self._source_name = None # name of source
    self._sub_name = None    # name of variable/method within source, if applicable
    self._obj = None         # reference to instance of the source, if applicable
    self._value = None       # used for fixed values
    self._growth_val = None  # used to grow the value year-by-year
    self._growth_mode = None # mode for growth (e.g. exponenetial, linear)

  def get_growth(self):
    return self._growth_val, self._growth_mode

  def get_source(self):
    """
      Accessor for the source type and name of this valued param
      @ In, None
      @ Out, type, str, identifier for the style of valued param
      @ Out, source name, str, name of the source
    """
    return self.type, self._source_name

  def get_values(self):
    """
      Accessor for value of the param.
      @ In, None
      @ Out, value, float, float value of parameter (if a valued_method)
    """
    return self._value

  def set_object(self, obj):
    """
      Setter for the evaluation target of this valued param (e.g., function, ARMA, etc).
      @ In, obj, isntance, evaluation target
      @ Out, None
    """
    self._obj = obj

  def read(self, comp_name, spec, mode, alias_dict=None):
    """
      Used to read valued param from XML input
      @ In, comp_name, str, name of component that this valued param will be attached to; only used for print messages
      @ In, spec, InputData params, input specifications
      @ In, mode, type of simulation calculation
      @ In, alias_dict, dict, optional, aliases to use for variable naming
    """
    # aliases get used to convert variable names, notably for the cashflow's "capacity"
    if alias_dict is None:
      alias_dict = {}
    # load this particular valued param type
    node, signal, request = self._load(comp_name, spec, mode, alias_dict)
    # if additional entities are needed to evaluate this param, note them now
    if request is not None:
      self.type, self._source_name = request
      self._sub_name = signal
    else:
      self.type = 'value'
      self._value = node.value
      signal = []
    return signal

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
    ret = None
    if self.type == 'value':
      value = {target_var: self._value}
    elif self.type == 'ARMA':
      value = self._evaluate_arma(inputs, target_var, aliases)
    elif self.type == 'variable':
      value = self._evaluate_variable(inputs, target_var, aliases)
    elif self.type == 'Function':
      # directly set the return, unlike the other types
      ret = self._evaluate_function(inputs, aliases)
    else:
      raise RuntimeError('Unrecognized data source:', self.type)
    # if the return dict wasn't auto-created, create it now
    if ret is None:
      ret = (value, inputs['meta'])
    return ret

  def _load(self, comp_name, item, mode, alias_dict):
    """
      Handles reading the values of valued parameters depending on the mode.
      @ In, comp_name, str, name of requesting component -> exclusively for error messages
      @ In, item, InputData, head XML with subparts that include options
      @ In, mode, str, mode of the case being run
      @ In, aliad_dict, dict, translation map for variable names
      @ Out, InputData, element to read from
      @ Out, signals, list, if additional signals needed then is a list of signals to be tracked
      @ Out, requests, tuple, node tag and text (corresponding to source type and name, respectively)
    """
    head_name = item.getName()
    err_msg = '\nFor "{}" in "{}", must provide exactly ONE of the following:'.format(head_name, comp_name) +\
                    '\n   - <ARMA>\n   - <Function>\n   - <variable>\n   '+\
                    '- Any combination of <fixed_value>, <sweep_values>, <opt_bounds>\n' +\
                    'No other option is currently implemented.'

    # handle growth factors
    growth = item.findFirst('growth')
    if growth is not None:
      self._growth_val = growth.value
      self._growth_mode = growth.parameterValues['mode']

    # find type of contents provided
    given = list(x.getName() for x in item.subparts if x.getName() != 'growth')

    has_vals = any(g in self.valued_methods for g in given)
    has_arma = any(g == 'ARMA' for g in given)
    has_func = any(g == 'Function' for g in given)
    has_vars = any(g == 'variable' for g in given)

    ## can't have more than one source option
    if not (has_vals + has_arma + has_func + has_vars == 1):
      raise IOError(err_msg)
    ## if arma, func, or var: check not multiply-defined
    if has_arma or has_func or has_vars:
      if len(given) > 1:
        raise IOError(err_msg)
      # otherwise, we only have the node we need, but we may need to store additional information
      ## namely, the signals, variables, and/or methods if ARMA or Function
      if has_arma:
        # we know what signals from the ARMA have been requested!
        signals = item.findFirst('ARMA').parameterValues['variable']
      elif has_func:
        # implementation check: we don't do growth with functions yet.
        if self._growth_val is not None:
          raise NotImplementedError('Currently <Function> and <growth> are not compatible. ' +
                                    'The same effect can be had by writing it directly into the Function '+
                                    'and accessing the "year" meta variable.')
        # function needs to know what member to use
        signals = item.subparts[0].parameterValues['method']
      elif has_vars:
        # implementation check: we don't do growth with variables yet.
        if self._growth_val is not None:
          raise NotImplementedError('Currently <variable> and <growth> are not compatible.')
        signals = item.subparts[0].value
        if signals in alias_dict:
          signals = alias_dict[signals]
      return item.subparts[0], signals, (item.subparts[0].getName(), item.subparts[0].value)
    ## by now, we only have the "valued" options
    # implementation check: we don't do growth with fixed vals yet.
    if self._growth_val is not None:
      raise NotImplementedError('Currently fixed, swept, and opt values and <growth> are not compatible.')
    # if fixed, take that.
    if 'fixed_value' in given:
      return item.subparts[given.index('fixed_value')], None, None
    # otherwise, it depends on the mode
    if mode == 'sweep':
      if 'sweep_values' in given:
        return item.subparts[given.index('sweep_values')], None, None
      else:
        raise IOError('For "{}" in "{}", no <sweep_values> given but in sweep mode! '.format(head_name, comp_name) +\
                      '\nPlease provide either <fixed_value> or <sweep_values>.')
    elif mode == 'opt':
      if 'opt_bounds' in given:
        return item.subparts[given.index('opt_bounds')], None, None
      else:
        raise IOError('For "{}" in "{}", no <opt_bounds> given but in sweep mode! '.format(head_name, comp_name) +\
                      '\nPlease provide either <fixed_value> or <opt_bounds>.')
    ## if we got here, then the rights nodes were not given!
    raise RuntimeError(err_msg)

  def _evaluate_arma(self, inputs, target_var, aliases):
    """
      Use an ARMA to evaluate this valued param
      @ In, inputs, dict, various variable-value pairs from the simulation
      @ In, target_var, str, intended key for the result of the evaluation
      @ In, aliases, dict, mapping for alternate naming if any
      @ Out, evaluate, dict, {var_name: var_value} evaluated
    """
    # find sampled value
    variable = self._sub_name # OLD info_dict['variable']
    # set "key" before checking aliases
    key = variable if not target_var else target_var
    if variable in aliases:
      variable = aliases[variable]
    t = inputs['t']
    year = inputs['meta']['EGRET']['sim_year_index']
    dim_map = inputs['raven_vars']['_indexMap'][0][variable]
    # TODO this is hard-coded to "Year", but so is the ARMA right now.
    if dim_map[0] == 'Year':
      selector = (year, t)
    else:
      selector = (t, year)
    #print('DEBUGG dim map:', dim_map)
    try:
      value = inputs['raven_vars'][variable][selector]
      #print('DEBUGG value:')
      #print(value)
      #vvvvvvv
    except KeyError as e:
      print('ERROR: variable "{}" not found among RAVEN variables!'.format(variable))
      raise e
    except IndexError as e:
      print('ERROR: requested index "{}" beyond length of evaluated ARMA "{}"!'.format(selector, self._source_name))
      raise e
    return {key: value} # float value

  def _evaluate_function(self, inputs, aliases):
    """
      Use a Python module to evaluate this valued param
      Note this isn't quite the same as leveraging RAVEN's Functions yet
       -> but it probably can be joined once the Functions are reworked.
      @ In, inputs, dict, various variable-value pairs from the simulation
      @ In, target_var, str, intended key for the result of the evaluation
      @ In, aliases, dict, mapping for alternate naming if any
      @ Out, balance, dict, {var_name: var_value} evaluated values (may be multiple)
      @ Out, meta, dict, potentially changed template-defined metadata
    """
    # TODO how to handle aliases????
    method = self._sub_name
    request = inputs.pop('request', None) # sometimes just the inputs will be used without the request, like in elasticity curves
    result = self._obj.evaluate(method, request, inputs)
    balance = result[0]
    meta = result[1]['meta']
    return balance, meta

  def _evaluate_variable(self, inputs, target_var, aliases):
    """
      Use a variable value from RAVEN or similar to evaluate this valued param.
      @ In, inputs, dict, various variable-value pairs from the simulation
      @ In, target_var, str, intended key for the result of the evaluation
      @ In, aliases, dict, mapping for alternate naming if any
      @ Out, evaluate, dict, {var_name: var_value} evaluated value
    """
    variable = self._sub_name
    # set "key" before checking aliases
    key = variable if not target_var else target_var
    if variable in aliases:
      variable = aliases[variable]
    try:
      val = inputs['raven_vars'][variable]
    except KeyError as e:
      print('ERROR: requested variable "{}" not found among RAVEN variables!'.format(variable))
      print('  -> Available:')
      for vn in inputs['raven_vars'].keys():
        print('      ', vn)
      raise e
    return {key: float(val)}


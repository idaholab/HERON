
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the Component entity.
"""
from __future__ import unicode_literals, print_function
import os
import sys
import _utils as hutils
framework_path = hutils.get_raven_loc()
sys.path.append(framework_path)
from utils import InputData, InputTypes

# class for potentially dynamically-evaluated quantities
class ValuedParam:
  """
    This class enables the identification of runtime-evaluated variables
    with a variety of sources (fixed values, parametric values, data histories, function
    evaluations, etc).

    # REFACTOR This should be split into the various ValuedParam types: value, sweep/opt, linear, function, ARMA
  """
  # these types represent values that do not need to be evaluated at run time, as they are determined.
  valued_methods = ['fixed_value', 'sweep_values', 'opt_bounds']

  @classmethod
  def get_input_specs(cls, name, disallowed=None):
    """
      Template for parameters that can take a scalar, an ARMA history, or a function
      @ In, name, string, name for spec (tag)
      @ In, disallowed, list(str), names of options not to be included
      @ Out, spec, InputData, value-based spec
    """
    if disallowed is None:
      disallowed = []
    spec = InputData.parameterInputFactory(name,
        descr=r"""This value can be taken from any \emph{one} of the sources described below.""")
    # for when the value is fixed (takes precedence over "sweep" and "opt") ...
    spec.addSub(InputData.parameterInputFactory('fixed_value', contentType=InputTypes.FloatType,
        descr=r"""indicates this value is a fixed value, given by the value of this node."""))
    # for when the value is parametric (only in "sweep" mode)
    spec.addSub(InputData.parameterInputFactory('sweep_values', contentType=InputTypes.FloatListType,
        descr=r"""indicates this value is parametric. The value of this node should contain
              two floats, representing the minimum and maximum values to sweep between."""))
    # for when the value is optimized (only in "min" or "max" mode)
    spec.addSub(InputData.parameterInputFactory('opt_bounds', contentType=InputTypes.FloatListType,
        descr=r"""indicates this value is an optimization variable. The optimization limits should
              be given in the value of this node as two float values."""))
    # for when the value is time-dependent and given by an ARMA
    arma = InputData.parameterInputFactory('ARMA', contentType=InputTypes.StringType,
        descr=r"""indicates that this value will be taken from synthetically-generated signals,
              which will be provided to the dispatch at run time by RAVEN from trained models. The value
              of this node should be the name of a synthetic history generator in the
              \xmlNode{DataGenerator} node.""")
    arma.addParam('variable', param_type=InputTypes.StringType,
        descr=r"""indicates which variable coming from the synthetic histories this value should be taken from.""")
    spec.addSub(arma)
    # for when the value comes from evaluating a function
    func = InputData.parameterInputFactory('Function', contentType=InputTypes.StringType,
        descr=r"""indicates this value should be taken from a Python function, as described
              in the \xmlNode{DataGenerators} node.""")
    func.addParam('method', param_type=InputTypes.StringType,
        descr=r"""the name of the \xmlNode{DataGenerator} from which this value should be taken.""")
    spec.addSub(func)
    # for when the value comes from another variable
    var = InputData.parameterInputFactory('variable', contentType=InputTypes.StringType,
        descr=r"""the name of the variable from the named function that will provide this value.""")
    spec.addSub(var)
    # for when linear coefficients are provided
    linear = InputData.parameterInputFactory('linear',
        descr=r"""indicates that linear coefficients will be provided to calculate values.""")
    lin_rate = InputData.parameterInputFactory('rate', contentType=InputTypes.FloatType,
        descr=r"""linear coefficient for the indicated \xmlAttr{resource}.""")
    lin_rate.addParam('resource', param_type=InputTypes.StringType,
        descr=r"""indicates the resource for which the linear transfer rate is being provided in this node.""")
    linear.addSub(lin_rate)
    spec.addSub(linear)
    # for when the result obtained needs to grow from year to year
    growth = InputData.parameterInputFactory('growth', contentType=InputTypes.FloatType,
        descr=r"""if this node is given, the value will be adjusted from cycle to cycle by the provided amount.""")
    growth_mode = InputTypes.makeEnumType('growthType', 'growthType', ['linear', 'exponential'])
    growth.addParam('mode', param_type=growth_mode, required=True,
        descr=r"""determines whether the growth factor should be taken as linear or exponential (compounding).""")
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
    self._coefficients = {}  # for (linear) coefficents, resource: coefficient

  def get_growth(self):
    """
      Obtains the yearly growth parameter and mode, if any
      @ In, None
      @ Out, val, float, growth value
      @ Out, mode, str, growth mode (linear or exponential)
    """
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
      @ Out, signal, list, signals needed to evaluate this ValuedParam at runtime
    """
    # aliases get used to convert variable names, notably for the cashflow's "capacity"
    if alias_dict is None:
      alias_dict = {}
    # load this particular valued param type
    typ, node, signal, request = self._load(comp_name, spec, mode, alias_dict)
    self.type = typ
    # if additional entities are needed to evaluate this param, note them now
    if typ == 'value':
      self._value = node.value
      signal = []
    elif typ == 'linear':
      signal = []
    else:
      self._source_name = request
      self._sub_name = signal
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
    elif self.type == 'linear':
      value = self._evaluate_linear(inputs, target_var, aliases)
    elif self.type == 'Function':
      # directly set the return, unlike the other types
      ret = self._evaluate_function(inputs, aliases)
    else:
      raise RuntimeError('Unrecognized data source:', self.type)
    # if the return dict wasn't auto-created, create it now
    if ret is None:
      ret = (value, inputs)
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
                    '\n   - <ARMA>\n   - <Function>\n   - <variable>\n   - <linear>\n   '+\
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
    has_linear = any(g == 'linear' for g in given)

    ## can't have more than one source option
    if not (has_vals + has_arma + has_func + has_vars + has_linear == 1):
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
      return item.subparts[0].getName(), item.subparts[0], signals, item.subparts[0].value

    if has_linear:
      # linear transfer coefficients are provided
      self._coefficients = {}
      # multiplicity
      if len(given) > 1:
        raise IOError(err_msg)
      for rate_node in item.findFirst('linear').findAll('rate'):
        resource = rate_node.parameterValues['resource']
        rate = rate_node.value
        self._coefficients[resource] = rate
      return 'linear', None, None, None

    ## by now, we only have the "valued" options
    # implementation check: we don't do growth with fixed vals yet.
    if self._growth_val is not None:
      raise NotImplementedError('Currently fixed, swept, and opt values and <growth> are not compatible.')
    # if fixed, take that.
    if 'fixed_value' in given:
      return 'value', item.subparts[given.index('fixed_value')], None, None
    # otherwise, it depends on the mode
    if mode == 'sweep':
      if 'sweep_values' in given:
        return 'value', item.subparts[given.index('sweep_values')], None, None
      else:
        raise IOError('For "{}" in "{}", no <sweep_values> given but in sweep mode! '.format(head_name, comp_name) +\
                      '\nPlease provide either <fixed_value> or <sweep_values>.')
    elif mode == 'opt':
      if 'opt_bounds' in given:
        return 'value', item.subparts[given.index('opt_bounds')], None, None
      else:
        raise IOError('For "{}" in "{}", no <opt_bounds> given but in "opt" mode! '.format(head_name, comp_name) +\
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
    t = inputs['HERON']['time_index']
    # there may not be a "Year" here, Year is handled by Manager
    # dim_map = inputs['HERON']['RAVEN_vars']['_indexMap'][0][variable]
    # if 'Year' in dim_map:
    #   # TODO this is hard-coded to "Year", but so is the ARMA right now.
    #   year = inputs['HERON']['active_index']['year']
    #   if dim_map[0] == 'Year':
    #     selector = (year, t)
    #   else:
    #     selector = (t, year)
    # else:
    selector = t
    try:
      value = inputs['HERON']['RAVEN_vars'][variable][selector]
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
    meta = result[1]
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

  def _evaluate_linear(self, inputs, target_var, aliases):
    """
      Evaluate a linear equation to determine value
      @ In, inputs, dict, various variable-value pairs from the simulation
      @ In, target_var, str, intended key for the result of the evaluation
      @ In, aliases, dict, mapping for alternate naming if any
      @ Out, balance, dict, {var_name: var_value} evaluated values (may be multiple)
    """
    # TODO how to handle aliases????
    if target_var not in self._coefficients:
      raise RuntimeError('Coefficient not defined for ValuedParam {}!'.format(self.name))
    # get the requested resource, and the requested amount
    req_res, req_amt = next(iter(inputs['request'].items()))
    # get the linear coefficient for the requested resource
    ## note this could be negative!
    req_rate = self._coefficients[req_res]
    balance = {req_res: req_amt}
    for res, rate in self._coefficients.items():
      # skip the requested resource, as we already handled it
      if res == req_res:
        continue
      # use the linear production ratio to how much of everything was used
      ## careful of the positive/negative signs!
      balance[res] = rate / req_rate * req_amt
    return balance


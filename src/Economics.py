"""
  Defines the Economics entity.
  Each component (or source?) can have one of these to describe its economics.
"""
from __future__ import unicode_literals, print_function
import os
import sys
from collections import defaultdict
import numpy as np
from ValuedParams import ValuedParam
import _utils as hutils
framework_path = hutils.get_raven_loc()
sys.path.append(framework_path)
from utils import InputData, xmlUtils,InputTypes


class CashFlowUser:
  """
    Base class for objects that want to access the functionality of the CashFlow objects.
    Generally this means the CashFlowUser will have an "economics" xml node used to define it,
    and will have a group of cash flows associated with it (e.g. a "component")

    In almost all cases, initialization methods should be called as part of the inheritor's method call.
  """
  @classmethod
  def get_input_specs(cls, spec):
    """
      Collects input specifications for this class.
      Note this needs to be called as part of an inheriting class's specification definition
      @ In, spec, InputData, specifications that need cash flow added to it
      @ Out, input_specs, InputData, specs
    """
    # this unit probably has some economics
    spec.addSub(CashFlowGroup.get_input_specs())
    return spec

  def __init__(self):
    """
      Constructor
      @ In, kwargs, dict, optional, arguments to pass to other constructors
      @ Out, None
    """
    self._economics = None # CashFlowGroup

  def read_input(self, specs):
    """
      Sets settings from input file
      @ In, specs, InputData params, input from user
      @ Out, None
    """
    self._economics = CashFlowGroup(self)
    self._economics.read_input(specs)

  def get_crossrefs(self):
    """
      Collect the required value entities needed for this component to function.
      @ In, None
      @ Out, crossrefs, dict, mapping of dictionaries with information about the entities required.
    """
    return self._economics.get_crossrefs()

  def set_crossrefs(self, refs):
    """
      Connect cross-reference material from other entities to the ValuedParams in this component.
      @ In, refs, dict, dictionary of entity information
      @ Out, None
    """
    self._economics.set_crossrefs(refs)

  def get_state_cost(self, activity, meta):
    """
      get the cost given particular activities (state) of the cash flow user
      @ In, raven_vars, dict, additional variables (presumably from raven) that might be needed
      @ In, meta, dict, further dictionary of information that might be needed
      XXX @ In, time, float, time at which cost needs to be evaluated
      @ Out, cost, dict, cost of activity as a breakdown
    """
    return self.get_economics().evaluate_cfs(activity, meta)

  def get_economics(self):
    """
      Accessor for economics.
      @ In, None
      @ Out, econ, CashFlowGroup, cash flows for this cash flow user
    """
    return self._economics


class CashFlowGroup:
  # Just a holder for multiple cash flows, and methods for doing stuff with them
  ##################
  # INITIALIZATION #
  ##################
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('economics', ordered=False, baseNode=None,
        descr=r"""this node is where all the economic information about this
              component is placed.""")
    specs.addSub(InputData.parameterInputFactory('lifetime', contentType=InputTypes.IntegerType,
        descr=r"""indicates the number of \emph{cycles} (often \emph{years}) this unit is expected
              to operate before replacement. Replacement is represented as overnight capital cost
              in the year the component is replaced."""))
    cf = CashFlow.get_input_specs()
    specs.addSub(cf)
    return specs

  def __init__(self, component):
    """
      Constructor.
      @ In, component, CashFlowUser instance, object to which this group belongs
      @ Out, None
    """
    self.name = component.name
    self._component = component # component this one
    self._lifetime = None # lifetime of the component
    self._cash_flows = []

  def read_input(self, source, xml=False):
    """
      Sets settings from input file
      @ In, source, InputData.ParameterInput, input from user
      @ In, xml, bool, if True then XML is passed in, not input data
      @ Out, None
    """
    print(' ... loading economics ...')
    # allow read_input argument to be either xml or input specs
    if xml:
      specs = self.get_input_specs()()
      specs.parseNode(source)
    else:
      specs = source
    # read in specs
    for item in specs.subparts:
      if item.getName() == 'lifetime':
        self._lifetime = item.value
      elif item.getName() == 'CashFlow':
        new = CashFlow(component=self._component)
        new.read_input(item)
        self._cash_flows.append(new)

  def get_crossrefs(self):
    """
      Provides a dictionary of the entities needed by this cashflow group to be evaluated
      @ In, None
      @ Out, crossreffs, dict, dictionary of crossreferences needed (see ValuedParams)
    """
    crossrefs = dict((cf, cf.get_crossrefs()) for cf in self._cash_flows)
    return crossrefs

  def set_crossrefs(self, refs):
    """
      Provides links to entities needed to evaluate this cash flow group.
      @ In, refs, dict, reference entities
      @ Out, None
    """
    for cf in list(refs.keys()):
      for try_match in self._cash_flows:
        if try_match == cf:
          try_match.set_crossrefs(refs.pop(try_match))
          break

  #######
  # API #
  #######
  def evaluate_cfs(self, activity, meta):
    """
      Calculates the incremental cost of a particular system configuration.
      @ In, activity, XArray.DataArray, array of driver-centric variable values
      @ In, raven_vars, dict, additional inputs from RAVEN call (or similar)
      @ In, meta, dict, additional user-defined meta
      XXX @ In, time, float, time at which cost needs to be evaluated
      @ Out, cost, dict, cash flow evaluations
    """
    # combine all cash flows into single cash flow evaluation
    cost = dict((cf.name, cf.evaluate_cost(activity, meta)) for cf in self.get_cashflows())
    return cost

  def get_cashflows(self):
    return self._cash_flows

  def get_component(self):
    """
      Return the cash flow user that owns this group
      @ In, None
      @ Out, component, CashFlowUser instance, owner
    """
    return self._component

  def get_lifetime(self):
    """
      Provides the lifetime of this cash flow user.
      @ In, None
      @ Out, lifetime, int, lifetime
    """
    return self._lifetime

  def check_if_finalized(self):
    return all(k.is_finalized() for k in self._cash_flows)

  def finalize(self, activity, raven_vars, meta, times=None):
    """
      Evaluate the parameters for member cash flows, and freeze values so they aren't changed again.
      @ In, activity, dict, mapping of variables to values (may be np.arrays)
      @ In,
    """
    info = {'raven_vars': raven_vars, 'meta': meta}
    for cf in self._cash_flows:
      cf.finalize(activity, info, times=times)

  def calculate_lifetime_cashflows(self):
    for cf in self._cash_flows:
      cf.calculate_lifetime_cashflow(self._lifetime)





class CashFlow:
  """
    Hold the economics for a single cash flow, C = m * a * (D/D')^x
    where:
      C is the cashflow ($)
      m is a scalar multiplier
      a is the value of the widget, based on the D' volume sold
      D is the amount of widgets sold
      D' is the nominal amount of widgets sold
      x is the scaling factor
  """
  ##################
  # INITIALIZATION #
  ##################
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    cf = InputData.parameterInputFactory('CashFlow')
    cf.descr = r"""node for defining a CashFlow for a particular Component. Note a CashFlow generally
               takes the form $C = \alpha \left(\frac{D}{\prime D}\right)\^x$, aggregated depending
               on the \xmlAttr{type}. For more information, see the CashFlow submodule of RAVEN."""

    cf.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name by which this CashFlow will be identified as part of this component. The
              general name is prefixed by the component name, such as ComponentName\|CashFlowName. """)
    cf_type_enum = InputTypes.makeEnumType('CFType', 'CFType', ['one-time', 'repeating'])
    cf.addParam('type', param_type=cf_type_enum, required=True,
        descr=r"""the type of CashFlow to calculate. \xmlString(one-time) is suitable for capital
              expenditure CashFlows, while \xmlString(repeating) is used for repeating costs such as
              operations and maintenance (fixed or variable), market sales, or similar.""")
    cf.addParam('taxable', param_type=InputTypes.BoolType, required=True,
        descr=r"""determines whether this CashFlow is taxed every cycle. """)
    cf.addParam('inflation', param_type=InputTypes.StringType, required=True,
        descr=r"""determines how inflation affects this CashFlow every cycle. See the CashFlow submodule
              of RAVEN.""")
    cf.addParam('mult_target', param_type=InputTypes.BoolType, required=True,
        descr=r"""indicates whether this parameter should be a target of the multiplication factor
              for NPV matching analyses.""")
    period_enum = InputTypes.makeEnumType('period_opts', 'period_opts', ['hour', 'year'])
    cf.addParam('period', param_type=period_enum, required=False,
        descr=r"""for a \xmlNode{CashFlow} with \xmlAttr{type} \xmlString{repeating}, indicates whether
              the CashFlow repeats every time step (\xmlString{hour}) or every cycle (\xmlString{year})).
              Generally, CashFlows such as fixed operations and maintenance costs are per-cycle, whereas
              variable costs such as fuel and maintenance as well as sales are repeated every time step.""")

    driver = ValuedParam.get_input_specs('driver')
    driver.descr = r"""indicates the main driver for this CashFlow, such as the number of units sold
                   or the size of the constructed unit. Corresponds to $D$ in the CashFlow equation."""
    cf.addSub(driver)

    reference_price = ValuedParam.get_input_specs('reference_price')
    reference_price.descr = r"""indicates the cash value of the reference number of units sold.
                            corresponds to $\alpha$ in the CashFlow equation. If \xmlNode{reference_driver}
                            is 1, then this is the price-per-unit for the CashFlow."""
    cf.addSub(reference_price)

    reference_driver = ValuedParam.get_input_specs('reference_driver')
    reference_driver.desecr = r"""determines the number of units sold to which the \xmlNode{reference_price}
                              refers. Corresponds to $\prime D$ in the CashFlow equation. """
    cf.addSub(reference_driver)

    x = ValuedParam.get_input_specs('scaling_factor_x')
    x.descr = r"""determines the scaling factor for this CashFlow. Corresponds to $x$ in the CashFlow
              equation. If $x$ is less than one, the per-unit price decreases as the units sold increases
              above the \xmlNode{reference_driver}, and vice versa."""
    cf.addSub(x)

    depreciate = InputData.parameterInputFactory('depreciate', contentType=InputTypes.IntegerType)
    depreciate.descr = r"""indicates the number of cycles over which this CashFlow should be depreciated.
                       Depreciation schemes are assumed to be MACRS and available cycles are listed
                       in the CashFlow submodule of RAVEN."""
    cf.addSub(depreciate)

    return cf

  def __init__(self, component):
    """
      Constructor
      @ In, component, CashFlowUser instance, cash flow user to which this cash flow belongs
      @ Out, None
    """
    # assert component is not None # TODO is this necessary? What if it's not a component-based cash flow?
    self._component = component # component instance to whom this cashflow belongs, if any
    # equation values
    self._driver = None       # ValuedParam "quantity produced", D
    self._alpha = None        # ValuedParam "price per produced", a
    self._reference = None    # ValuedParam "where price is accurate", D'
    self._scale = None        # ValuedParam "economy of scale", x
    # other params
    self.name = None          # base name of cash flow
    self._type = None         # needed? one-time, yearly, repeating
    self._taxable = None      # apply tax or not
    self._inflation = None    # apply inflation or not
    self._mult_target = None  # not clear
    self._depreciate = None
    self._period = None       # period for recurring cash flows
    # other members
    self._signals = set()     # variable values needed for this cash flow
    self._crossrefs = defaultdict(dict)


  def read_input(self, item):

    """
      Sets settings from input file
      @ In, item, InputData.ParameterInput, parsed specs from user
      @ Out, None
    """
    self.name = item.parameterValues['name']
    print(' ... ... loading cash flow "{}"'.format(self.name))
    # handle type directly here momentarily
    self._taxable = item.parameterValues['taxable']
    self._inflation = item.parameterValues['inflation']
    self._mult_target = item.parameterValues['mult_target']
    self._type = item.parameterValues['type']
    self._period = item.parameterValues.get('period', 'hour')
    # the remainder of the entries are ValuedParams, so they'll be evaluated as-needed
    for sub in item.subparts:
      if sub.getName() == 'driver':
        self._set_valued_param('_driver', sub)
      elif sub.getName() == 'reference_price':
        self._set_valued_param('_alpha', sub)
      elif sub.getName() == 'reference_driver':
        self._set_valued_param('_reference', sub)
      elif sub.getName() == 'scaling_factor_x':
        self._set_valued_param('_scale', sub)
      elif sub.getName() == 'depreciate':
        self._depreciate = sub.value

      else:
        raise IOError('Unrecognized "CashFlow" node: "{}"'.format(sub.getName()))

    # driver is required!
    if self._driver is None:
      raise IOError('No <driver> node provided for CashFlow {}!'.format(self.name))
    if self._alpha is None:
      raise IOError('No <reference_price> node provided for CashFlow {}!'.format(self.name))

    # defaults
    var_names = ['_reference', '_scale']
    for name in var_names:
      if getattr(self, name) is None:
        # TODO raise a warning?
        self._set_fixed_param(name, 1)

  # Not none set it to default 1
  def get_period(self):
    return self._period

  def _set_fixed_param(self, name, value):
    vp = ValuedParam(name)
    vp.type = 'value'
    vp._value = value # TODO directly accessing private member!
    setattr(self, name, vp)

  def _set_valued_param(self, name, spec):
    """
      Utilitly method to set ValuedParam members via reading input specifications.
      @ In, name, str, member variable name (e.g. self.<name>)
      @ In, spec, InputData params, input parameters
      @ Out, None
    """
    vp = ValuedParam(name)
    signal = vp.read('CashFlow \'{}\''.format(self.name), spec, None) # TODO what "mode" to use?
    self._signals.update(signal)
    self._crossrefs[name] = vp
    # alias: redirect "capacity" variable
    if vp.type == 'variable' and vp._sub_name == 'capacity':
      vp = self._component.get_capacity_param()
    setattr(self, name, vp)

  def get_alpha_extension(self):
    """ creates multiplier for the valued shape the alpha cashflow parameter should be in """
    life = self._component.get_economics().get_lifetime()
    if self._type == 'one-time':
      ext = np.zeros(life+1, dtype=float)
      ext[0] = 1.0
    elif self._type == 'repeating':
      ext = np.ones(life+1, dtype=float)
      ext[0] = 0.0
    else:
      raise NotImplementedError('type is: {}'.format(self._type))
    return ext

  def get_crossrefs(self):
    """
      Accessor for cross-referenced entities needed by this cashflow.
      @ In, None
      @ Out, crossrefs, dict, cross-referenced requirements dictionary
    """
    return self._crossrefs

  def set_crossrefs(self, refs):
    """
      Setter for cross-referenced entities needed by this cashflow.
      @ In, refs, dict, cross referenced entities
      @ Out, None
    """
    for attr, obj in refs.items():
      valued_param = self._crossrefs[attr]
      valued_param.set_object(obj)

  def evaluate_cost(self, activity, values_dict):
    """
      Evaluates cost of a particular scenario provided by "activity".
      @ In, activity, pandas.Series, multi-indexed array of scenario activities
      @ In, values_dict, dict, additional values that may be needed to evaluate cost
      @ In, t, int, time index at which cost should be evaluated
      @ Out, cost, float, cost of activity
    """
    # note this method gets called a LOT, so speedups here are quite effective
    # "activity" is a pandas series with production levels -> example from HERON case
    # build aliases
    aliases = {} # currently unused, but mechanism left in place
    #aliases['capacity'] = '{}_capacity'.format(self._component.name)
    # for now, add the activity to the dictionary
    #res_vals = activity.to_dict() # TODO slow, speed this up
    # if 'HERON' not in values_dict['meta']:
    #   values_dict['meta']['HERON'] = {}
    values_dict['HERON']['activity'] = activity
    params = self.calculate_params(values_dict)
    return params['cost']

  def calculate_params(self, values_dict):
    """
      Calculates the value of the cash flow parameters.
      @ In, values_dict, dict, mapping from simulation variable names to their values (as floats or numpy arrays)
      @ Out, params, dict, dictionary of parameters mapped to values including the cost
    """
    Dp = float(self._reference.evaluate(values_dict, target_var='reference_driver')[0]['reference_driver'])
    x = float(self._scale.evaluate(values_dict, target_var='scaling_factor_x')[0]['scaling_factor_x'])
    ## a and D need to be filled as time dependent
    ### TODO messes with Pyomo, just do one time at a time for now
    # a = np.zeros(T)
    # D = np.zeros(T)
    # for i, t in enumerate(times):
    #   values_dict['t'] = t
    #   a[i] = self._alpha.evaluate(values_dict, target_var='reference_price', aliases=aliases)[0]['reference_price']#[0]
    #   D[i] = self._driver.evaluate(values_dict, target_var='driver', aliases=aliases)[0]['driver']
    # if aggregate:
    #   # parameters might be time-dependent, so aggregate them appropriately
    #   if T > 1:
    #     ## "alpha" should be the average price
    #     a = float(a.mean())
    #     ## "D" should be the total amount produced
    #     D = float(D.sum())
    #   elif T == 1:
    #     a = float(a)
    #     D = float(D)
    #   else:
    #     raise RuntimeError('Requested time stamps were empty!')
    ### TODO pyomo safe
    #values_dict['t'] = times[0]
    a = self._alpha.evaluate(values_dict, target_var='reference_price')[0]['reference_price']
    D = self._driver.evaluate(values_dict, target_var='driver')[0]['driver']
    cost = a * (D / Dp) ** x
    params = {'alpha': a, 'driver': D, 'ref_driver': Dp, 'scaling': x, 'cost': cost} # TODO float(cost) except in pyomo it's not a float
    return params

  def get_cashflow_params(self, values_dict, aliases, dispatches, years):
    """ creates a param dict for initializing a CashFlows.CashFlow """
    # OLD
    params = {'name': self.name,
              'reference': vals_dict['ref_driver'],
              'driver': vals_dict['driver'],
              'alpha': vals_dict['alpha'],
              'X': vals_dict['scaling'],
              'mult_target': self._mult_target,
              'inflation': self._inflation,
              'multiply': None,
              'tax': self._taxable,
              }
    return params

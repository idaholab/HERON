
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the Economics entity.
  Each component (or source?) can have one of these to describe its economics.
"""
from __future__ import unicode_literals, print_function
import sys
from collections import defaultdict
import numpy as np
from HERON.src import ValuedParams
from HERON.src.ValuedParamHandler import ValuedParamHandler
import HERON.src._utils as hutils
try:
  import ravenframework
except ModuleNotFoundError:
  framework_path = hutils.get_raven_loc()
  sys.path.append(framework_path)
from ravenframework.utils import InputData, xmlUtils,InputTypes


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
      @ Out, spec, InputData, specs
    """
    # this unit probably has some economics
    spec.addSub(CashFlowGroup.get_input_specs())
    return spec

  def __init__(self):
    """
      Constructor
      @ In, None
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

  def get_cashflows(self):
    """
      Getter.
      @ In, None
      @ Out, cashflow, list, cash flows for this cashflow user (ordered)
    """
    return self._economics.get_cashflows()

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

  def get_state_cost(self, activity, meta, marginal=False):
    """
      get the cost given particular activities (state) of the cash flow user
      @ In, raven_vars, dict, additional variables (presumably from raven) that might be needed
      @ In, meta, dict, further dictionary of information that might be needed
      @ In, marginal, bool, optional, if True then only get marginal costs
      @ Out, cost, dict, cost of activity as a breakdown
    """
    return self.get_economics().evaluate_cfs(activity, meta, marginal=marginal)

  def get_economics(self):
    """
      Accessor for economics.
      @ In, None
      @ Out, econ, CashFlowGroup, cash flows for this cash flow user
    """
    return self._economics


class CashFlowGroup:
  """
    Just a holder for multiple cash flows, and methods for doing stuff with them
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
      @ Out, crossrefs, dict, dictionary of crossreferences needed (see ValuedParams)
    """
    crossrefs = dict((cf, cf.get_crossrefs()) for cf in self._cash_flows)
    return crossrefs

  def set_crossrefs(self, refs):
    """
      Provides links to entities needed to evaluate this cash flow group.
      @ In, refs, dict, reference entities
      @ Out, None
    """
    # set up pointers
    for cf in list(refs.keys()):
      for try_match in self._cash_flows:
        if try_match == cf:
          try_match.set_crossrefs(refs.pop(try_match))
          break
      else:
        cf.set_crossrefs({})
    # perform checks


  #######
  # API #
  #######
  def evaluate_cfs(self, activity, meta, marginal=False):
    """
      Calculates the incremental cost of a particular system configuration.
      @ In, activity, XArray.DataArray, array of driver-centric variable values
      @ In, meta, dict, additional user-defined meta
      @ In, marginal, bool, optional, if True then only get marginal cashflows (e.g. recurring hourly)
      @ Out, cost, dict, cash flow evaluations
    """
    # combine all cash flows into single cash flow evaluation
    if marginal:
      # FIXME assuming 'year' is the only non-marginal value
      # FIXME why is it "repeating" and not "Recurring"?
      cost = dict((cf.name, cf.evaluate_cost(activity, meta))
                    for cf in self.get_cashflows()
                    if (cf.get_type() == 'repeating' and cf.get_period() != 'year'))
    else:
      cost = dict((cf.name, cf.evaluate_cost(activity, meta))
                    for cf in self.get_cashflows())
    return cost

  def get_cashflows(self):
    """
      Getter.
      @ In, None
      @ Out, cashflow, list, cash flows for this cashflow group (ordered)
    """
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
    """
      Check finalization status of cashflows for this group.
      @ In, None
      @ Out, finalized, bool, True if all are finalized
    """
    return all(k.is_finalized() for k in self._cash_flows)

  def finalize(self, activity, raven_vars, meta, times=None):
    """
      Evaluate the parameters for member cash flows, and freeze values so they aren't changed again.
      @ In, activity, dict, mapping of variables to values (may be np.arrays)
      @ In, raven_vars, dict, TODO part of meta! Consolidate!
      @ In, times, list, optional, times to finalize values for
      @ Out, None
    """
    info = {'raven_vars': raven_vars, 'meta': meta}
    for cf in self._cash_flows:
      cf.finalize(activity, info, times=times)

  def calculate_lifetime_cashflows(self):
    """
      Passthrough to CashFlow method of the same name.
      @ In, None
      @ Out, None
    """
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
    cf.description = r"""node for defining a CashFlow for a particular Component. This HERON
               CashFlow will be used to generate a TEAL CashFlow from RAVEN's TEAL plugin. Note a CashFlow generally
               takes the form $C = \alpha \left(\frac{D}{D'}\right)^x$, aggregated depending
               on the \xmlAttr{type}. For more information, see the TEAL plugin for RAVEN."""

    cf.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name by which this CashFlow will be identified as part of this component. The
              general name is prefixed by the component name, such as ComponentName$\vert$CashFlowName. """)
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
    cf.addParam('mult_target', param_type=InputTypes.BoolType, required=False,
        descr=r"""<DEPRECATED> indicates whether this parameter should be a target of the multiplication factor
              for NPV matching analyses. This should now be specified within the desired Cash Flow: under the
              ``reference price'' node with a ``levelized cost'' subnode. """)
    period_enum = InputTypes.makeEnumType('period_opts', 'period_opts', ['hour', 'year'])
    cf.addParam('period', param_type=period_enum, required=False,
        descr=r"""for a \xmlNode{CashFlow} with \xmlAttr{type} \xmlString{repeating}, indicates whether
              the CashFlow repeats every time step (\xmlString{hour}) or every cycle (\xmlString{year})).
              Generally, CashFlows such as fixed operations and maintenance costs are per-cycle, whereas
              variable costs such as fuel and maintenance as well as sales are repeated every time step.""")

    descr = r"""indicates the main driver for this CashFlow, such as the number of units sold
            or the size of the constructed unit. Corresponds to $D$ in the CashFlow equation."""
    driver = ValuedParams.factory.make_input_specs('driver', descr=descr, kind='post-dispatch')
    cf.addSub(driver)

    descr = r"""indicates the cash value of the reference number of units sold.
            corresponds to $\alpha$ in the CashFlow equation. If \xmlNode{reference_driver}
            is 1, then this is the price-per-unit for the CashFlow."""
    reference_price = ValuedParams.factory.make_input_specs('reference_price', descr=descr, kind='post-dispatch')
    levelized_cost = InputData.parameterInputFactory('levelized_cost', strictMode=True,
                                            descr=r"""indicates whether HERON and TEAL are meant to solve for the
                                            levelized price related to this cashflow.""")
    reference_price.addSub(levelized_cost)
    cf.addSub(reference_price)

    descr = r"""determines the number of units sold to which the \xmlNode{reference_price}
            refers. Corresponds to $\prime D$ in the CashFlow equation. """
    reference_driver = ValuedParams.factory.make_input_specs('reference_driver', descr=descr, kind='post-dispatch')
    cf.addSub(reference_driver)

    descr = r"""determines the scaling factor for this CashFlow. Corresponds to $x$ in the CashFlow
            equation. If $x$ is less than one, the per-unit price decreases as the units sold increases
            above the \xmlNode{reference_driver}, and vice versa."""
    x = ValuedParams.factory.make_input_specs('scaling_factor_x', descr=descr, kind='post-dispatch')
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
    # handle type directly here momentarily
    self._taxable = item.parameterValues['taxable']
    self._inflation = item.parameterValues['inflation']
    self._type = item.parameterValues['type']
    self._period = item.parameterValues.get('period', 'hour')
    # the remainder of the entries are ValuedParams, so they'll be evaluated as-needed
    for sub in item.subparts:
      if sub.getName() == 'driver':
        self._set_valued_param('_driver', sub)
      elif sub.getName() == 'reference_price':
        price_is_levelized = self.set_reference_price(sub) # setting "_alpha" here
      elif sub.getName() == 'reference_driver':
        self._set_valued_param('_reference', sub)
      elif sub.getName() == 'scaling_factor_x':
        self._set_valued_param('_scale', sub)
      elif sub.getName() == 'depreciate':
        self._depreciate = sub.value

      else:
        raise IOError(f'Unrecognized "CashFlow" node: {sub.getName()}')

    # resolve levelized cost
    self._mult_target = price_is_levelized
    # user asked to find Time Invariant levelized cost
    if self._alpha is None and price_is_levelized:
      self._set_fixed_param('_alpha', 1)

    # driver is required!
    if self._driver is None:
      raise IOError(f'No <driver> node provided for CashFlow {self.name}!')

    # defaults
    var_names = ['_reference', '_scale']
    for name in var_names:
      if getattr(self, name) is None:
        # TODO raise a warning?
        self._set_fixed_param(name, 1)

  def set_reference_price(self, node):
    """
      Sets the reference_price attribute based on given ValuedParam or if Levelized Cost
      @ In, node, InputParams.ParameterInput, reference_price head node
      @ Out, price_is_levelized, bool, are we computing levelized cost for this cashflow?
    """
    levelized_cost = False
    for sub in node.subparts:
      if sub.name == 'levelized_cost':
        levelized_cost = True
        __ = node.popSub('levelized_cost')

    try:
      self._set_valued_param('_alpha', node)
    except AttributeError as e:
      if levelized_cost:
        self._set_fixed_param('_alpha', 1)
      else:
        raise IOError(f'No <reference_price> node provided for CashFlow {self.name}!') from e
    price_is_levelized = bool(levelized_cost)
    return price_is_levelized


  # Not none set it to default 1
  def get_period(self):
    """
      Getter for Recurring cashflow period type.
      @ In, None
      @ Out, period, str, 'hourly' or 'yearly'
    """
    return self._period

  def _set_fixed_param(self, name, value):
    """
      Fixes a ValuedParam to have a constant value
      @ In, name, str, name of member to store on "self"
      @ In, value, float, value to set for ValuedParam
      @ Out, None
    """
    vp = ValuedParamHandler(name)
    vp.set_const_VP(value)
    setattr(self, name, vp)

  def _set_valued_param(self, name, spec):
    """
      Utilitly method to set ValuedParam members via reading input specifications.
      @ In, name, str, member variable name (e.g. self.<name>)
      @ In, spec, InputData params, input parameters
      @ Out, None
    """
    vp = ValuedParamHandler(name)
    signal = vp.read(f'CashFlow \'{self.name}\'', spec, None) # TODO what "mode" to use?
    self._signals.update(signal)
    self._crossrefs[name] = vp
    # standard alias: redirect "capacity" variable
    if isinstance(vp, ValuedParams.factory.returnClass('variable')) and vp.get_raven_var() == 'capacity':
      #NOTE: we are assuming here that capacity_factors are only applied in dispatch and
      # are not a variable in the outer optimization.
      vp = self._component.get_capacity_param()
    setattr(self, name, vp)

  def get_alpha_extension(self):
    """
      creates multiplier for the valued shape the alpha cashflow parameter should be in
      @ In, None,
      @ Out, ext, multiplier for "alpha" values based on CashFlow type
    """
    life = self._component.get_economics().get_lifetime()
    if self._type == 'one-time':
      ext = np.zeros(life+1, dtype=float)
      ext[0] = 1.0
    elif self._type == 'repeating':
      ext = np.ones(life+1, dtype=float)
      ext[0] = 0.0
    else:
      raise NotImplementedError(f'type is: {self._type}')
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
    # set up pointers
    for attr, obj in refs.items():
      valued_param = self._crossrefs[attr]
      valued_param.set_object(obj)
    # check on VP setup
    for attr, vp in self._crossrefs.items():
      vp.crosscheck(self._component.get_interaction())

  def evaluate_cost(self, activity, values_dict):
    """
      Evaluates cost of a particular scenario provided by "activity".
      @ In, activity, pandas.Series, multi-indexed array of scenario activities
      @ In, values_dict, dict, additional values that may be needed to evaluate cost
      @ In, t, int, time index at which cost should be evaluated
      @ Out, cost, float, cost of activity
    """
    # note this method gets called a LOT, so speedups here are quite effective
    # add the activity to the dictionary
    values_dict['HERON']['activity'] = activity
    params = self.calculate_params(values_dict)
    return params['cost']

  def calculate_params(self, values_dict):
    """
      Calculates the value of the cash flow parameters.
      @ In, values_dict, dict, mapping from simulation variable names to their values (as floats or numpy arrays)
      @ Out, params, dict, dictionary of parameters mapped to values including the cost
    """
    # TODO maybe don't cast these as floats, as they could be symbolic expressions (seems unlikely)
    Dp = float(self._reference.evaluate(values_dict, target_var='reference_driver')[0]['reference_driver'])
    x = float(self._scale.evaluate(values_dict, target_var='scaling_factor_x')[0]['scaling_factor_x'])
    a = self._alpha.evaluate(values_dict, target_var='reference_price')[0]['reference_price']
    D = self._driver.evaluate(values_dict, target_var='driver')[0]['driver']
    cost = a * (D / Dp) ** x
    params = {'alpha': a, 'driver': D, 'ref_driver': Dp, 'scaling': x, 'cost': cost} # TODO float(cost) except in pyomo it's not a float
    return params

  #######
  # API #
  #######
  def get_price(self):
    """
      Getter for Cashflow Price
      @ In, None
      @ Out, alpha, ValuedParam, valued param for the cash flow price
    """
    return self._alpha

  def get_driver(self):
    """
      Getter for Cashflow Driver
      @ In, None
      @ Out, driver, ValuedParam, valued param for the cash flow driver
    """
    return self._driver

  def get_reference(self):
    """
      Getter for Cashflow Reference Driver
      @ In, None
      @ Out, reference, ValuedParam, valued param for the cash flow reference driver
    """
    return self._reference

  def get_scale(self):
    """
      Getter for Cashflow Scale
      @ In, None
      @ Out, scale, ValuedParam, valued param for the cash flow economy of scale
    """
    return self._scale

  def get_type(self):
    """
      Getter for Cashflow Type
      @ In, None
      @ Out, type, str, one-time, yearly, repeating
    """
    return self._type

  def get_depreciation(self):
    """
      Getter for Cashflow depreciation
      @ In, None
      @ Out, depreciate, int or None
    """
    return self._depreciate

  def is_taxable(self):
    """
      Getter for Cashflow taxable boolean
      @ In, None
      @ Out, taxable, bool, is cashflow taxable?
    """
    return self._taxable

  def is_inflation(self):
    """
      Getter for Cashflow inflation boolean
      @ In, None
      @ Out, inflation, bool, is inflation applied to cashflow?
    """
    return self._inflation

  def is_mult_target(self):
    """
      Getter for Cashflow mult_target boolean
      @ In, None
      @ Out, taxable, bool, is cashflow a multiplier target?
    """
    return self._mult_target

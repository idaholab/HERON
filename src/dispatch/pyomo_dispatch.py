"""
  pyomo-based dispatch strategy
"""

import os
import sys
import time as time_mod
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
import pyomo.environ as pyo

# allows pyomo to solve on threaded processes
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

from .Dispatcher import Dispatcher
try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils


class Pyomo(Dispatcher):
  """
    Dispatches using rolling windows in Pyomo
  """
  naming_template = {'comp prod': '{comp}|{res}|prod',
                     'comp transfer': '{comp}|{res}|trans',
                     'comp max': '{comp}|{res}|max',
                     'comp ramp up': '{comp}|{res}|rampup',
                     'comp ramp down': '{comp}|{res}|rampdown',
                     'conservation': '{res}|consv',
                    }

  ### INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('Dispatcher', ordered=False, baseNode=None)
    # TODO specific for pyomo dispatcher
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'PyomoDispatcher' # identifying name
    self._window_len = 24         # time window length to dispatch at a time # FIXME user input

  def read_input(self, specs):
    """
      Read in input specifications.
      @ In, specs, RAVEN InputData, specifications
      @ Out, None
    """
    print('DEBUGG specs:', specs)

  ### API
  def dispatch(self, case, components, sources, variables):
    """
      Performs dispatch.
      @ In, case, HERON Case, Case that this dispatch is part of
      @ In, components, list, HERON components available to the dispatch
      @ In, sources, list, HERON source (placeholders) for signals
      @ In, variables, dict, additional variables passed through
      @ Out, disp, DispatchScenario, resulting dispatch
    """
    time = np.arange(100) # FIXME
    resources = sorted(list(hutils.get_all_resources(components))) # list of all active resources
    # TODO rolling window?
    start_index = 0
    final_index = len(time)
    # TODO make dispatch scenario? How to store results?
    while start_index < final_index:
      end_index = start_index + self._window_len
      print('DEBUGG starting window {} to {}'.format(start_index, end_index))
      if end_index > final_index:
        end_index = final_index
      start = time_mod.time()
      subdisp = self.dispatch_window(start_index, end_index,
                                     case, components, sources, resources,
                                     variables)
      end = time_mod.time()
      print('DEBUGG solve time: {} s'.format(end-start))
      import pprint
      pprint.pprint(subdisp)
      # TODO update dispatch
      CRASHME
      start_index = end_index

  ### INTERNAL
  def dispatch_window(self, start_index, end_index,
                      case, components, sources, resources,
                      variables):
    # build the Pyomo model
    # TODO abstract this model as much as possible BEFORE, then concrete initialization per window
    m = pyo.ConcreteModel()
    # indices
    C = np.arange(0, len(components), dtype=int) # indexes component
    R = np.arange(0, len(resources), dtype=int) # indexes resources
    T = np.arange(start_index, end_index, dtype=int) # indexes resources
    m.C = pyo.Set(initialize=C)
    m.R = pyo.Set(initialize=R)
    m.T = pyo.Set(initialize=T)
    m.resource_index_map = {} # maps the resource to its index WITHIN APPLICABLE components (sparse matrix)
                              #   e.g. component: {resource: local index}, ... etc}
    # properties
    m.Case = case
    m.Components = components
    # constraints and variables
    for comp in components:
      # NOTE: "fixed" components could hypothetically be treated differently
      ## however, in order for the "production" variable for components to be treatable as the
      ## same as other production variables, we create components with limitation
      ## lowerbound == upperbound == capacity (just for "fixed" dispatch components)
      prod_name = self._create_production(m, comp) # variables
      self._create_capacity(m, comp, prod_name)    # capacity constraints
      self._create_transfer(m, comp, prod_name)    # transfer functions (constraints)
      # ramp rates TODO ## INCLUDING previous-time boundary condition TODO
    self._create_conservation(m, resources) # conservation of resources (e.g. production == consumption)
    self._create_objective(m) # objective
    # solve
    self._debug_pyomo_print(m)
    soln = pyo.SolverFactory('cbc').solve(m)
    soln.write() # DEBUGG
    self._debug_print_soln(m) # DEBUGG
    # return dict of numpy arrays
    result = self._retrieve_solution(m)
    return result

  ### PYOMO Element Constructors
  def _create_fixed(self, m, comp):
    """ TODO """
    # FIXME didn't work, so set minimum == maximum constraint
    xxxxxxx
    name = comp.name
    r_i_map = dict((res, r) for r, res in enumerate(comp.get_resources()))
    m.resource_index_map[name] = r_i_map
    res_indexer = pyo.Set(initialize=range(len(r_i_map)))
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    cap = comp.get_capacity(None, None, None, None)[0][cap_res] # value of capacity limit (units of governing resource)
    # size for usage
    cap *= np.ones(len(m.T))
    data = np.empty((len(r_i_map), len(m.T)))
    ratios = self._get_transfer_coeffs(m, comp)
    for res, r in r_i_map.items():
      if res == cap_res:
        data[r][:] = cap
      else:
        data[r][:] = cap * ratios[res]
    #indexer = partial(self._get_fixed_activity, data)
    #param = pyo.Param(res_indexer, m.T, initialize=indexer)
    setattr(m, '{c}_production'.format(c=name), data) # TODO I don't like this, but it works.

  def _create_production(self, m, comp):
    """ TODO """
    name = comp.name
    # map resources to their index FOR THIS COMPONENT (not global)
    m.resource_index_map[name] = dict((res, r) for r, res in enumerate(comp.get_resources()))
    # create pyomo indexer for this component's resources
    res_indexer = pyo.Set(initialize=range(len(m.resource_index_map[name])))
    setattr(m, '{c}_res_index_map'.format(c=name), res_indexer)
    # production variable depends on resource, time
    # # TODO if transfer function is linear, isn't this technically redundant? Maybe only need one resource ...
    ## Method 1: set variable bounds directly --> not working! why??
    #lower, upper, domain = self._get_prod_bounds(comp)
    #prod = pyo.Var(res_indexer, m.T, bounds=(lower, upper)) #within=domain,
    ## Method 2: set capacity as a seperate constraint
    prod = pyo.Var(res_indexer, m.T, initialize=0)
    prod_name = '{c}_production'.format(c=name)
    setattr(m, prod_name, prod)
    return prod_name

  def _create_capacity(self, m, comp, prod_name):
    """ TODO """
    name = comp.name
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    r = m.resource_index_map[name][cap_res] # production index of the governing resource
    # production is always lower than capacity
    ## NOTE get_capacity returns (data, meta) and data is dict
    cap = comp.get_capacity(None, None, None, None)[0][cap_res] # value of capacity limit (units of governing resource)
    rule = partial(self._capacity_rule, prod_name, r, cap)
    constr = pyo.Constraint(m.T, rule=rule)
    setattr(m, '{c}_{r}_capacity_constr'.format(c=name, r=cap_res), constr)
    # minimum production
    if comp.is_dispatchable() == 'fixed':
      minimum = cap
    else:
      minimum = 0 #  -> for now just use 0, but fix this! XXX
    rule = partial(self._min_prod_rule, prod_name, r, cap, minimum)
    constr = pyo.Constraint(m.T, rule=rule)
    setattr(m, '{c}_{r}_minprod_constr'.format(c=name, r=cap_res), constr)

  def _create_transfer(self, m, comp, prod_name):
    """ TODO """
    name = comp.name
    # transfer functions
    # e.g. 2A + 3B -> 1C + 2E
    # get linear coefficients
    ratios = self._get_transfer_coeffs(m, comp)
    ref_r, ref_name, _ = ratios.pop('__reference', (None, None, None))
    for resource, ratio in ratios.items():
      r = m.resource_index_map[name][resource]
      rule_name = '{c}_{r}_{fr}_transfer'.format(c=name, r=resource, fr=ref_name)
      rule = partial(self._transfer_rule, ratio, r, ref_r, prod_name) # XXX
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, rule_name, constr)

  def _create_conservation(self, m, resources):
    """ TODO """
    for res, resource in enumerate(resources):
      rule = partial(self._conservation_rule, resource) #lambda m, c, t: abs(np.sum(m.Production[c, res, t])) <=1e-14 # TODO zero tolerance value?
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, '{r}_conservation'.format(r=resource), constr)

  def _create_objective(self, m):
    """ TODO """
    ## cashflow eval
    m.obj = pyo.Objective(rule=self._cashflow_rule)

  ### UTILITIES for general use
  def _get_prod_bounds(self, comp):
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    maximum = comp.get_capacity(None, None, None, None)[0][cap_res]
    # TODO minimum!
    # producing or consuming the defining resource?
    if maximum > 0:
      return 0, maximum, pyo.NonNegativeReals
    else:
      return maximum, 0, pyo.NonPositiveReals

  def _get_transfer_coeffs(self, m, comp):
    name = comp.name
    transfer = comp.get_interaction().get_transfer()  # get the transfer ValuedParam, if any
    if transfer is None:
      return {}
    coeffs = transfer._coefficients # linear transfer coefficients, dict as {resource: coeff}, SIGNS MATTER
    # it's all about ratios -> store as ratio of resource / first resource (arbitrary)
    first_r = None
    first_name = None
    first_coef = None
    ratios = {}
    for resource, coef in coeffs.items():
      r = m.resource_index_map[name][resource]
      if first_r is None:
        # set up nominal resource to compare to
        first_r = r
        first_name = resource
        first_coef = coef
        ratios['__reference'] = (first_r, first_name, first_coef)
        continue
      ratio = coef / first_coef
      ratios[resource] = ratio
    return ratios

  def _retrieve_solution(self, m):
    """ TODO """
    #result = np.full((len(m.T), len(m.Components), len(m.Resources)), None)
    result = {} # {component: {resource: production}}
    for comp in m.Components:
      prod = getattr(m, '{n}_production'.format(n=comp.name))
      result[comp.name] = {}
      for res, comp_r in m.resource_index_map[comp.name].items():
        #global_r = m.Resources.index(res)
        result[comp.name][res] = np.fromiter((prod[comp_r, t].value for t in m.T), dtype=float, count=len(m.T))
        #result[:, c, global_r] = list(prod[comp_r, t] for t in m.T) # TODO can we extract the vector?
    return result

  ### RULES for partial function calls
  def _capacity_rule(self, prod_name, r, cap, m, t):
    """ Constructs capacity constraints. TODO"""
    prod = getattr(m, prod_name)
    if cap > 0:
      return prod[r, t] <= cap
    else:
      return prod[r, t] >= cap

  def _cashflow_rule(self, m):
    total = 0 # sum of cashflows
    activity = self._retrieve_solution(m)
    for comp in m.Components:
      name = comp.name
      indexer = m.resource_index_map[comp.name]
      df = pd.DataFrame.from_dict(activity[name], dtype=float)
      print('DEBUGG df:', df)
      for t in m.T:
        value = comp.get_incremental_cost(df, {}, {}, 0) # TODO activity, raven_vars, meta, t
        print('DEBUGG value:', comp.name, t, value)
        iiiiiii
        total += sum(value.values())
    return - total # minimization sense

  def _conservation_rule(self, res, m, t):
    """ Constructs conservation constraints. TODO """
    balance = 0
    for comp_name, res_dict in m.resource_index_map.items():
      if res in res_dict:
        balance += getattr(m, '{c}_production'.format(c=comp_name))[res_dict[res], t]
    return balance == 0 # TODO tol?

  def _min_prod_rule(self, prod_name, r, cap, minimum, m, t):
    prod = getattr(m, prod_name)
    if cap > 0:
      return prod[r, t] >= minimum
    else:
      return prod[r, t] <= minimum

  def _transfer_rule(self, ratio, r, ref_r, prod_name, m, t):
    """ Constructs transfer function constraints TODO"""
    prod = getattr(m, prod_name)
    return prod[r, t] == prod[ref_r, t] * ratio # TODO tolerance??

  def _get_fixed_activity(self, array, m, r, t):
    """ TODO """
    # FIXME this didn't work at all, it evaluated as a partial not a valued index ...
    return array[r, t]

  ### DEBUG
  def _debug_pyomo_print(self, m):
    """ TODO """
    print('/' + '='*80)
    print('DEBUGG model pieces:')
    print('  -> objective:')
    print('     ', m.obj.pprint())
    print('  -> variables:')
    for var in m.component_objects(pyo.Var):
      print('     ', var.pprint())
    print('  -> constraints:')
    for constr in m.component_objects(pyo.Constraint):
      print('     ', constr.pprint())
    print('\\' + '='*80)
    print('')

  def _debug_print_soln(self, m):
    """ TODO """
    print('DEBUGG solution:')
    for c, comp in enumerate(m.Components):
      name = comp.name
      print('  component:', c, name)
      for res, r in m.resource_index_map[name].items():
        print('    resource:', r, res)
        for t, time in enumerate(m.T):
          prod = getattr(m, '{n}_production'.format(n=name))
          print('      time:', t, time, prod[r, t].value)

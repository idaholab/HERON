
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  pyomo-based dispatch strategy
"""

import os
import sys
import time as time_mod
from functools import partial
import platform

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from utils import InputData, InputTypes

# allows pyomo to solve on threaded processes
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

from .Dispatcher import Dispatcher
from .DispatchState import DispatchState, NumpyState
try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils

# Choose solver; CBC is a great choice unless we're on Windows
if platform.system() == 'Windows':
  SOLVER = 'glpk'
else:
  SOLVER = 'cbc'


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
    specs = InputData.parameterInputFactory('pyomo', ordered=False, baseNode=None)
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
  def dispatch(self, case, components, sources, meta):
    """
      Performs dispatch.
      @ In, case, HERON Case, Case that this dispatch is part of
      @ In, components, list, HERON components available to the dispatch
      @ In, sources, list, HERON source (placeholders) for signals
      @ In, meta, dict, additional variables passed through
      @ Out, disp, DispatchScenario, resulting dispatch
    """
    t_start, t_end, t_num = self.get_time_discr()
    time = np.linspace(t_start, t_end, t_num) # Note we don't care about segment/cluster here
    resources = sorted(list(hutils.get_all_resources(components))) # list of all active resources
    # pre-build results structure
    ## we can use NumpyState here so we don't need to worry about a Pyomo model object
    dispatch = NumpyState()# dict((comp.name, dict((res, np.zeros(len(time))) for res in comp.get_resources())) for comp in components)
    dispatch.initialize(components, meta['HERON']['resource_indexer'], time)
    # rolling window
    start_index = 0
    final_index = len(time)
    # TODO window overlap!  ( )[ ] -> (   [  )   ]
    while start_index < final_index:
      end_index = start_index + self._window_len
      if end_index > final_index:
        end_index = final_index # + 1?
      specific_time = time[start_index:end_index]
      print('DEBUGG starting window {} to {}'.format(start_index, end_index))
      start = time_mod.time()
      subdisp = self.dispatch_window(specific_time,
                                     case, components, sources, resources,
                                     meta)
      end = time_mod.time()
      print('DEBUGG solve time: {} s'.format(end-start))
      # store result in corresponding part of dispatch
      for comp in components:
        for res, values in subdisp[comp.name].items():
          dispatch.set_activity_vector(comp, res, start_index, end_index, values)
      start_index = end_index
    return dispatch

  ### INTERNAL
  def dispatch_window(self, time,
                      case, components, sources, resources,
                      meta):
    """
      Dispatches one part of a rolling window.
      @ In, time, np.array, value of time to evaluate
      @ In, case, HERON Case, Case that this dispatch is part of
      @ In, components, list, HERON components available to the dispatch
      @ In, sources, list, HERON source (placeholders) for signals
      @ In, resources, list, sorted list of all resources in problem
      @ In, meta, dict, additional variables passed through
      @ Out, result, dict, results of window dispatch
    """
    # build the Pyomo model
    # TODO abstract this model as much as possible BEFORE, then concrete initialization per window
    m = pyo.ConcreteModel()
    # indices
    C = np.arange(0, len(components), dtype=int) # indexes component
    R = np.arange(0, len(resources), dtype=int) # indexes resources
    # T = np.arange(start_index, end_index, dtype=int) # indexes resources
    T = np.arange(0, len(time), dtype=int) # indexes resources
    m.C = pyo.Set(initialize=C)
    m.R = pyo.Set(initialize=R)
    m.T = pyo.Set(initialize=T)
    m.Times = time
    m.resource_index_map = meta['HERON']['resource_indexer'] # maps the resource to its index WITHIN APPLICABLE components (sparse matrix)
                                                             #   e.g. component: {resource: local index}, ... etc}
    # properties
    m.Case = case
    m.Components = components
    m.Activity = PyomoState()
    m.Activity.initialize(m.Components, m.resource_index_map, m.Times, m)
    # constraints and variables
    for comp in components:
      # NOTE: "fixed" components could hypothetically be treated differently
      ## however, in order for the "production" variable for components to be treatable as the
      ## same as other production variables, we create components with limitation
      ## lowerbound == upperbound == capacity (just for "fixed" dispatch components)
      prod_name = self._create_production(m, comp) # variables
      self._create_capacity(m, comp, prod_name, meta)    # capacity constraints
      self._create_transfer(m, comp, prod_name)    # transfer functions (constraints)
      # ramp rates TODO ## INCLUDING previous-time boundary condition TODO
    self._create_conservation(m, resources, meta) # conservation of resources (e.g. production == consumption)
    self._create_objective(meta, m) # objective
    # start a solution search
    done_and_checked = False
    attempts = 0
    # DEBUGG show variables, bounds
    # self._debug_pyomo_print(m)
    while not done_and_checked:
      attempts += 1
      print(f'DEBUGG solve attempt {attempts} ...:')
      # solve
      soln = pyo.SolverFactory(SOLVER).solve(m)
      # check solve status
      if soln.solver.status == SolverStatus.ok and soln.solver.termination_condition == TerminationCondition.optimal:
        print('DEBUGG ... solve was successful!')
      else:
        print('DEBUGG ... solve was unsuccessful!')
        print('DEBUGG ... status:', soln.solver.status)
        print('DEBUGG ... termination:', soln.solver.termination_condition)
        self._debug_pyomo_print(m)
        raise RuntimeError
      # try validating
      print('DEBUGG ... validating ...')
      validation_errs = self.validate(m.Components, m.Activity, m.Times, meta)
      if validation_errs:
        done_and_checked = False
        print('DEBUGG ... validation concerns raised:')
        for e in validation_errs:
          print('DEBUGG ... ... Time {t} ({time}) Component "{c}" Resource "{r}": {m}'
                .format(t=e['time_index'],
                        time=e['time'],
                        c=e['component'].name,
                        r=e['resource'],
                        m=e['msg']))
          self._create_production_limit(m, e)
        # go back and solve again
        # raise NotImplementedError('Validation failed, but idk how to handle that yet')
      else:
        print('DEBUGG Solve successful and no validation concerns raised.')
        done_and_checked = True
      if attempts > 100:
        raise RuntimeError('Exceeded validation attempt limit!')
    # soln.write() # DEBUGG
    # self._debug_print_soln(m) # DEBUGG
    # return dict of numpy arrays
    result = self._retrieve_solution(m)
    return result

  ### PYOMO Element Constructors
  def _create_production_limit(self, m, validation):
    """
      Creates pyomo production constraint given validation errors
      @ In, m, pyo.ConcreteModel, associated model
      @ In, validation, dict, information from Validator about limit violation
      @ Out, None
    """
    # TODO could validator write a symbolic expression on request? That'd be sweet.
    comp = validation['component']
    resource = validation['resource']
    r = m.resource_index_map[comp][resource]
    t = validation['time_index']
    limit = validation['limit']
    limits = np.zeros(len(m.Times))
    limits[t] = limit
    limit_type = validation['limit_type']
    prod_name = f'{comp.name}_production'
    rule = partial(self._prod_limit_rule, prod_name, r, limits, limit_type, t)
    constr = pyo.Constraint(rule=rule)
    counter = 1
    name_template = '{c}_{r}_{t}_vld_limit_constr_{{i}}'.format(c=comp.name,
                                                                r=resource,
                                                                t=t)
    # make sure we get a unique name for this constraint
    name = name_template.format(i=counter)
    while getattr(m, name, None) is not None:
      counter += 1
      name = name_template.format(i=counter)
    setattr(m, name, constr)
    print(f'DEBUGG added validation constraint "{name}"')

  def _create_production(self, m, comp):
    """
      Creates production pyomo variable object for a component
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make production variables for
      @ Out, prod_name, str, name of production variable
    """
    name = comp.name
    # create pyomo indexer for this component's resources
    res_indexer = pyo.Set(initialize=range(len(m.resource_index_map[comp])))
    setattr(m, f'{name}_res_index_map', res_indexer)
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

  def _create_capacity(self, m, comp, prod_name, meta):
    """
      Creates pyomo capacity constraints
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make variables for
      @ In, prod_name, str, name of production variable
      @ In, meta, dict, additional state information
      @ Out, None
    """
    name = comp.name
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    r = m.resource_index_map[comp][cap_res] # production index of the governing resource
    # production is always lower than capacity
    ## NOTE get_capacity returns (data, meta) and data is dict
    ## TODO does this work with, e.g., ARMA-based capacities?
    ### -> "time" is stored on "m" and could be used to correctly evaluate the capacity
    caps = []
    mins = []
    for t, time in enumerate(m.Times):
      meta['HERON']['time_index'] = t
      cap = comp.get_capacity(meta)[0][cap_res] # value of capacity limit (units of governing resource)
      caps.append(cap)
      minimum = comp.get_minimum(meta)[0][cap_res]
      # minimum production
      if (comp.is_dispatchable() == 'fixed') or (minimum == cap):
        minimum = cap
        # initialize values so there's no boundary errors
        var = getattr(m, prod_name)
        values = var.get_values()
        for k in values:
          values[k] = cap
        var.set_values(values)
      mins.append(minimum)
    # capacity
    rule = partial(self._capacity_rule, prod_name, r, caps)
    constr = pyo.Constraint(m.T, rule=rule)
    setattr(m, '{c}_{r}_capacity_constr'.format(c=name, r=cap_res), constr)
    # minimum
    rule = partial(self._min_prod_rule, prod_name, r, caps, mins)
    constr = pyo.Constraint(m.T, rule=rule)
    setattr(m, '{c}_{r}_minprod_constr'.format(c=name, r=cap_res), constr)

  def _create_transfer(self, m, comp, prod_name):
    """
      Creates pyomo transfer function constraints
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make variables for
      @ In, prod_name, str, name of production variable
      @ Out, None
    """
    name = comp.name
    # transfer functions
    # e.g. 2A + 3B -> 1C + 2E
    # get linear coefficients
    # TODO this could also take a transfer function from an external Python function assuming
    #    we're careful about how the expression-vs-float gets used
    #    and figure out how to handle multiple ins, multiple outs
    ratios = self._get_transfer_coeffs(m, comp)
    ref_r, ref_name, _ = ratios.pop('__reference', (None, None, None))
    for resource, ratio in ratios.items():
      r = m.resource_index_map[comp][resource]
      rule_name = '{c}_{r}_{fr}_transfer'.format(c=name, r=resource, fr=ref_name)
      rule = partial(self._transfer_rule, ratio, r, ref_r, prod_name) # XXX
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, rule_name, constr)

  def _create_conservation(self, m, resources, meta):
    """
      Creates pyomo conservation constraints
      @ In, m, pyo.ConcreteModel, associated model
      @ In, resources, list, list of resources in problem
      @ In, meta, dict, dictionary of state variables
      @ Out, None
    """
    for res, resource in enumerate(resources):
      rule = partial(self._conservation_rule, meta, resource)
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, '{r}_conservation'.format(r=resource), constr)

  def _create_objective(self, meta, m):
    """
      Creates pyomo objective function
      @ In, meta, dict, additional variables to pass through
      @ In, m, pyo.ConcreteModel, associated model
      @ Out, None
    """
    ## cashflow eval
    rule = partial(self._cashflow_rule, meta)
    m.obj = pyo.Objective(rule=rule, sense=pyo.maximize)

  ### UTILITIES for general use
  def _get_prod_bounds(self, comp):
    """
      Determines the production limits of the given component
      @ In, comp, HERON component, component to get bounds of
      @ Out, (min, max, domain), float/float/pyomo domain, limits and domain of variables
    """
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    maximum = comp.get_capacity(None, None, None, None)[0][cap_res]
    # TODO minimum!
    # producing or consuming the defining resource?
    if maximum > 0:
      return 0, maximum, pyo.NonNegativeReals
    else:
      return maximum, 0, pyo.NonPositiveReals

  def _get_transfer_coeffs(self, m, comp):
    """
      Obtains transfer function ratios (assuming Linear ValuedParams)
      Form: 1A + 3B -> 2C + 4D
      Ratios are calculated with respect to first resource listed, so e.g. B = 3/1 * A
      TODO I think we can handle general external functions, maybe?
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON component, component to get coefficients of
      @ Out, ratios, dict, ratios of transfer function variables
    """
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
      r = m.resource_index_map[comp][resource]
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
    """
      Extracts solution from Pyomo optimization
      @ In, m, pyo.ConcreteModel, associated (solved) model
      @ Out, result, dict, {comp: {resource: [production], etc}, etc}
    """
    result = {} # {component: {resource: production}}
    for comp in m.Components:
      prod = getattr(m, '{n}_production'.format(n=comp.name))
      result[comp.name] = {}
      for res, comp_r in m.resource_index_map[comp].items():
        result[comp.name][res] = np.fromiter((prod[comp_r, t].value for t in m.T), dtype=float, count=len(m.T))
    return result

  ### RULES for partial function calls
  # these get called using "functools.partial" to make Pyomo constraints, vars, objectives, etc

  def _capacity_rule(self, prod_name, r, caps, m, t):
    """
      Constructs pyomo capacity constraints.
      @ In, prod_name, str, name of production variable
      @ In, r, int, index of resource for capacity constraining
      @ In, caps, list(float), value to constrain resource at in time
      @ In, m, pyo.ConcreteModel, associated model
      @ In, t, int, time index for capacity rule
    """
    kind = 'lower' if min(caps) < 0 else 'upper'
    return self._prod_limit_rule(prod_name, r, caps, kind, t, m)

  def _prod_limit_rule(self, prod_name, r, limits, kind, t, m):
    """
      Constructs pyomo production constraints.
      @ In, prod_name, str, name of production variable
      @ In, r, int, index of resource for capacity constraining
      @ In, limits, list(float), values in time at which to constrain resource production
      @ In, kind, str, either 'upper' or 'lower' for limiting production
      @ In, t, int, time index for production rule (NOTE not pyomo index, rather fixed index)
      @ In, m, pyo.ConcreteModel, associated model
    """
    prod = getattr(m, prod_name)
    if kind == 'lower':
      # production must exceed value
      return prod[r, t] >= limits[t]
    elif kind == 'upper':
      return prod[r, t] <= limits[t]
    else:
      raise TypeError('Unrecognized production limit "kind":', kind)

  def _cashflow_rule(self, meta, m):
    """
      Objective function rule.
      @ In, meta, dict, additional variable passthrough
      @ In, m, pyo.ConcreteModel, associated model
      @ Out, total, float, evaluation of cost
    """
    activity = m.Activity # dict((comp, getattr(m, f"{comp.name}_production")) for comp in m.Components)
    state_args = {'valued': False}
    total = self._compute_cashflows(m.Components, activity, m.Times, meta, state_args=state_args)
    return total

  def _conservation_rule(self, meta, res, m, t):
    """
      Constructs conservation constraints.
      @ In, meta, dict, dictionary of state variables
      @ In, res, str, name of resource
      @ In, m, pyo.ConcreteModel, associated model
      @ In, t, int, index of time variable
      @ Out, conservation, bool, balance check
    """
    balance = 0 # sum of production rates, which needs to be zero
    for comp, res_dict in m.resource_index_map.items():
      if res in res_dict:
        # activity information depends on if storage or component
        var = getattr(m, f'{comp.name}_production')
        r = res_dict[res]
        if comp.get_interaction().is_type('Storage'):
          # Storages store LEVELS not ACTIVITIES, so calculate activity
          # Production rate for storage defined as R_k = (L_{k+1} - L_k) / dt
          if t > 0:
            previous = var[r, t-1]
            dt = m.Times[t] - m.Times[t-1]
          else:
            # FIXME check this with a variety of ValuedParams
            previous = comp.get_interaction().get_initial_level(meta)
            dt = m.Times[1] - m.Times[0]
          new = var[r, t]
          production = -1 * (new - previous) / dt # swap sign b/c negative is absorbing, positive is emitting
        else:
          # TODO move to this? balance += m._activity.get_activity(comp, res, t)
          production = var[r, t]
        balance += production
    return balance == 0 # TODO tol?

  def _min_prod_rule(self, prod_name, r, caps, minimums, m, t):
    """
      Constructs minimum production constraint
      @ In, prod_name, str, name of production variable
      @ In, r, int, index of resource for capacity constraining
      @ In, caps, list(float), capacity(t) value for component
      @ In, minimums, list(float), minimum allowable production in time
      @ In, m, pyo.ConcreteModel, associated model
      @ In, t, int, index of time variable
      @ Out, minimum, bool, min check
    """
    prod = getattr(m, prod_name)
    # negative capacity means consuming instead of producing
    if max(caps) > 0:
      return prod[r, t] >= minimums[t]
    else:
      return prod[r, t] <= minimums[t]

  def _transfer_rule(self, ratio, r, ref_r, prod_name, m, t):
    """
      Constructs transfer function constraints
      @ In, ratio, float, ratio for resource to nominal first resource
      @ In, r, int, index of transfer resource
      @ In, ref_r, int, index of reference resource
      @ In, prod_name, str, name of production variable
      @ In, m, pyo.ConcreteModel, associated model
      @ In, t, int, index of time variable
      @ Out, transfer, bool, transfer ratio check
    """
    prod = getattr(m, prod_name)
    return prod[r, t] == prod[ref_r, t] * ratio # TODO tolerance??

  ### DEBUG
  def _debug_pyomo_print(self, m):
    """
      Prints the setup pieces of the Pyomo model
      @ In, m, pyo.ConcreteModel, model to interrogate
      @ Out, None
    """
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
    """
      Prints the solution from the Pyomo model
      @ In, m, pyo.ConcreteModel, model to interrogate
      @ Out, None
    """
    print('*'*80)
    print('DEBUGG solution:')
    print('  objective value:', m.obj())
    for c, comp in enumerate(m.Components):
      name = comp.name
      print('  component:', c, name)
      for res, r in m.resource_index_map[comp].items():
        print('    resource:', r, res)
        for t, time_index in enumerate(m.T):
          prod = getattr(m, '{n}_production'.format(n=name))
          print('      time:', t, time_index, prod[r, time_index].value)
    print('*'*80)





# DispatchState for Pyomo dispatcher
class PyomoState(DispatchState):
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    DispatchState.__init__(self)
    self._model = None # Pyomo model object

  def initialize(self, components, resources_map, times, model):
    """
      Connect information about this State to other objects
      @ In, components, list, HERON components
      @ In, resources_map, dict, map of component names to resources used
      @ In, times, np.array, values of "time" this state represents
      @ In, model, pyomo.Model, associated model for this state
      @ Out, None
    """
    DispatchState.initialize(self, components, resources_map, times)
    self._model = model

  def get_activity_indexed(self, comp, r, t, valued=True, **kwargs):
    """
      Getter for activity level.
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, r, int, index of resource to retrieve (as given by meta[HERON][resource_indexer])
      @ In, t, int, index of time at which activity should be provided
      @ In, valued, bool, optional, if True then get float value instead of pyomo expression
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    prod = getattr(self._model, f'{comp.name}_production')[r, t]
    if valued:
      return prod()
    return prod

  def set_activity_indexed(self, comp, r, t, value, valued=False):
    raise NotImplementedError

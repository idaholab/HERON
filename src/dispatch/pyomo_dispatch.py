
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  pyomo-based dispatch strategy
"""

import os
import sys
import time as time_mod
import platform
from itertools import compress
import pprint

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from ravenframework.utils import InputData, InputTypes

# allows pyomo to solve on threaded processes
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
from pyomo.common.errors import ApplicationError

from .Dispatcher import Dispatcher
from .DispatchState import DispatchState, NumpyState
try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils

# Choose solver; CBC is a great choice unless we're on Windows
if platform.system() == 'Windows':
  SOLVERS = ['glpk', 'cbc', 'ipopt']
else:
  SOLVERS = ['cbc', 'glpk', 'ipopt']


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
    specs = InputData.parameterInputFactory('pyomo', ordered=False, baseNode=None,
        descr=r"""The \texttt{pyomo} dispatcher uses analytic modeling and rolling windows to
        solve dispatch optimization with perfect information via the pyomo optimization library.""")
    specs.addSub(InputData.parameterInputFactory('rolling_window_length', contentType=InputTypes.IntegerType,
        descr=r"""Sets the length of the rolling window that the Pyomo optimization algorithm
        uses to break down histories. Longer window lengths will minimize boundary effects, such as
        nonoptimal storage dispatch, at the cost of slower optimization solves.
        Note that if the rolling window results in a window of length 1 (such as at the end of a history),
        this can cause problems for pyomo.
        \default{24}"""))
    specs.addSub(InputData.parameterInputFactory('debug_mode', contentType=InputTypes.BoolType,
        descr=r"""Enables additional printing in the pyomo dispatcher. Highly discouraged for production runs.
        \default{False}."""))
    specs.addSub(InputData.parameterInputFactory('solver', contentType=InputTypes.StringType,
        descr=r"""Indicates which solver should be used by pyomo. Options depend on individual installation.
        \default{'glpk' for Windows, 'cbc' otherwise}."""))
    # TODO specific for pyomo dispatcher
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'PyomoDispatcher' # identifying name
    self.debug_mode = False       # whether to print additional information
    self._window_len = 24         # time window length to dispatch at a time # FIXME user input
    self._solver = None           # overwrite option for solver
    self._picard_limit = 10       # iterative solve limit

  def read_input(self, specs):
    """
      Read in input specifications.
      @ In, specs, RAVEN InputData, specifications
      @ Out, None
    """
    super().read_input(specs)

    window_len_node = specs.findFirst('rolling_window_length')
    if window_len_node is not None:
      self._window_len = window_len_node.value

    debug_node = specs.findFirst('debug_mode')
    if debug_node is not None:
      self.debug_mode = debug_node.value

    solver_node = specs.findFirst('solver')
    if solver_node is not None:
      self._solver = solver_node.value

    if self._solver is None:
      solvers_to_check = SOLVERS
    else:
      solvers_to_check = [self._solver]
    # check solver exists
    for solver in solvers_to_check:
      self._solver = solver
      found_solver = True
      try:
        if not pyo.SolverFactory(self._solver).available():
          found_solver = False
        else:
          break
      except ApplicationError:
        found_solver = False
    # NOTE: we probably need a consistent way to test and check viable solvers,
    # maybe through a unit test that mimics the model setup here. For now, I assume
    # that anything that shows as not available or starts with an underscore is not
    # viable, and it will crash if the solver can't solve our kinds of models.
    # This should only come up if the user is specifically requesting a solver, though,
    # the default glpk and cbc are tested.
    if not found_solver:
      all_options = pyo.SolverFactory._cls.keys() # TODO shorten to list of tested options?
      solver_filter = []
      for op in all_options:
        if op.startswith('_'): # These don't seem like legitimate options, based on testing
          solver_filter.append(False)
          continue
        try:
          solver_filter.append(pyo.SolverFactory(op).available())
        except (ApplicationError, NameError, ImportError):
          solver_filter.append(False)
      available = list(compress(all_options, solver_filter))
      msg = f'Requested solver "{self._solver}" was not found for pyomo dispatcher!'
      msg += f' Options MAY include: {available}'
      raise RuntimeError(msg)


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
        end_index = final_index
      if end_index - start_index == 1:
        # TODO custom error raise for catching in DispatchManager?
        raise IOError("A rolling window of length 1 was requested, but this causes crashes in pyomo. " +
                      "Change the length of the rolling window to avoid length 1 histories.")
      specific_time = time[start_index:end_index]
      print('DEBUGG starting window {} to {}'.format(start_index, end_index))
      start = time_mod.time()
      # set initial storage levels
      initial_levels = {}
      for comp in components:
        if comp.get_interaction().is_type('Storage'):
          if start_index == 0:
            initial_levels[comp] = comp.get_interaction().get_initial_level(meta)
          else:
            initial_levels[comp] = subdisp[comp.name]['level'][comp.get_interaction().get_resource()][-1]
      # allow for converging solution iteratively
      converged = False
      conv_counter = 0
      previous = None
      while not converged:
        conv_counter += 1
        if conv_counter > self._picard_limit:
          break
        # dispatch
        subdisp = self.dispatch_window(specific_time, start_index,
                                      case, components, sources, resources,
                                      initial_levels, meta)
        # do we need a convergence criteria? Check now.
        if self.needs_convergence(components):
          print(f'DEBUGG iteratively solving window, iteration {conv_counter}/{self._picard_limit} ...')
          converged = self.check_converged(subdisp, previous, components)
          previous = subdisp
        else:
          converged = True

      end = time_mod.time()
      print('DEBUGG solve time: {} s'.format(end-start))
      # store result in corresponding part of dispatch
      for comp in components:
        for tag in comp.get_tracking_vars():
          for res, values in subdisp[comp.name][tag].items():
            dispatch.set_activity_vector(comp, res, values, tracker=tag, start_idx=start_index, end_idx=end_index)
      start_index = end_index
    return dispatch

  ### INTERNAL
  def dispatch_window(self, time, time_offset,
                      case, components, sources, resources,
                      initial_storage, meta):
    """
      Dispatches one part of a rolling window.
      @ In, time, np.array, value of time to evaluate
      @ In, time_offset, int, offset of the time index in the greater history
      @ In, case, HERON Case, Case that this dispatch is part of
      @ In, components, list, HERON components available to the dispatch
      @ In, sources, list, HERON source (placeholders) for signals
      @ In, resources, list, sorted list of all resources in problem
      @ In, initial_storage, dict, initial storage levels if any
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
    dt = m.Times[1] - m.Times[0] # TODO assumes consistent step sizing
    m.time_offset = time_offset
    m.resource_index_map = meta['HERON']['resource_indexer'] # maps the resource to its index WITHIN APPLICABLE components (sparse matrix)
                                                             #   e.g. component: {resource: local index}, ... etc}
    # properties
    m.Case = case
    m.Components = components
    m.Activity = PyomoState()
    m.Activity.initialize(m.Components, m.resource_index_map, m.Times, m)
    # constraints and variables
    for comp in components:
      # components using a governing strategy (not opt) are Parameters, not Variables
      # TODO should this come BEFORE or AFTER each dispatch opt solve?
      # -> responsive or proactive?
      intr = comp.get_interaction()
      if intr.is_governed():
        meta['request'] = {'component': comp, 'time': time}
        activity = intr.get_strategy().evaluate(meta)[0]['level']
        if intr.is_type('Storage'):
          self._create_production_param(m, comp, activity, tag='level')
          # set up "activity" rates (change in level over time, plus efficiency)
          rte2 = comp.get_sqrt_RTE() # square root of the round-trip efficiency
          L  = len(activity)
          deltas = np.zeros(L)
          deltas[1:] = activity[1:] - activity[:-1]
          deltas[0] = activity[0] - intr.get_initial_level(meta)
          # rate of charge
          # change sign, since increasing level means absorbing energy from system
          # also scale by RTE, since to get level increase you have to over-absorb
          charge = np.zeros(L)
          charge_mask = np.where(deltas > 0)
          charge[charge_mask] = - deltas[charge_mask] / dt / rte2
          self._create_production_param(m, comp, charge, tag='charge')
          # rate of discharge
          # change sign, since decreasing level means emitting energy into system
          # also scale by RTE, since level decrease yields less to system
          discharge = np.zeros(L)
          discharge_mask = np.where(deltas < 0)
          discharge[discharge_mask] = - deltas[discharge_mask] / dt * rte2
          self._create_production_param(m, comp, discharge, tag='discharge')
        else:
          self._create_production_param(m, comp, activity)
        continue
      # NOTE: "fixed" components could hypothetically be treated differently
      ## however, in order for the "production" variable for components to be treatable as the
      ## same as other production variables, we create components with limitation
      ## lowerbound == upperbound == capacity (just for "fixed" dispatch components)
      if intr.is_type('Storage'):
        self._create_storage(m, comp, initial_storage, meta)
      else:
        self._create_production(m, comp, meta) # variables
    self._create_conservation(m, resources, initial_storage, meta) # conservation of resources (e.g. production == consumption)
    self._create_objective(meta, m) # objective
    # start a solution search
    done_and_checked = False
    attempts = 0
    # DEBUGG show variables, bounds
    if self.debug_mode:
      self._debug_pyomo_print(m)
    while not done_and_checked:
      attempts += 1
      print(f'DEBUGG solve attempt {attempts} ...:')
      # solve
      # TODO someday if we want to give user access to options, we can add them to this dict. For now, no options.
      solve_options = {}
      soln = pyo.SolverFactory(self._solver).solve(m, options=solve_options)
      # check solve status
      if soln.solver.status == SolverStatus.ok and soln.solver.termination_condition == TerminationCondition.optimal:
        print('DEBUGG ... solve was successful!')
      else:
        print('DEBUGG ... solve was unsuccessful!')
        print('DEBUGG ... status:', soln.solver.status)
        print('DEBUGG ... termination:', soln.solver.termination_condition)
        self._debug_pyomo_print(m)
        print('Resource Map:')
        pprint.pprint(m.resource_index_map)
        raise RuntimeError(f"Solve was unsuccessful! Status: {soln.solver.status} Termination: {soln.solver.termination_condition}")
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
    if self.debug_mode:
      soln.write()
      self._debug_print_soln(m)
    # return dict of numpy arrays
    result = self._retrieve_solution(m)
    return result

  def check_converged(self, new, old, components):
    """
      Checks convergence of consecutive dispatch solves
      @ In, new, dict, results of dispatch # TODO should this be the model rather than dict?
      @ In, old, dict, results of previous dispatch
      @ In, components, list, HERON component list
      @ Out, converged, bool, True if convergence is met
    """
    tol = 1e-4 # TODO user option
    if old is None:
      return False
    converged = True
    for comp in components:
      intr = comp.get_interaction()
      name = comp.name
      tracker = comp.get_tracking_vars()[0]
      if intr.is_governed(): # by "is_governed" we mean "isn't optimized in pyomo"
        # check activity L2 norm as a differ
        # TODO this may be specific to storage right now
        res = intr.get_resource()
        scale = np.max(old[name][tracker][res])
        diff = np.linalg.norm(new[name][tracker][res] - old[name][tracker][res]) / (scale if scale != 0 else 1)
        if diff > tol:
          converged = False
    return converged

  def needs_convergence(self, components):
    """
      Determines whether the current setup needs convergence to solve.
      @ In, components, list, HERON component list
      @ Out, needs_convergence, bool, True if iteration is needed
    """
    for comp in components:
      intr = comp.get_interaction()
      # storages with a prescribed strategy MAY need iteration
      if intr.is_governed():
        return True
    # if we get here, no iteration is needed
    return False


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
    rule = lambda mod: self._prod_limit_rule(prod_name, r, limits, limit_type, t, mod)
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

  def _create_production_param(self, m, comp, values, tag=None):
    """
      Creates production pyomo fixed parameter object for a component
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make production variables for
      @ In, values, np.array(float), values to set for param
      @ In, tag, str, optional, if not None then name will be component_[tag]
      @ Out, prod_name, str, name of production variable
    """
    name = comp.name
    if tag is None:
      tag = 'production'
    # create pyomo indexer for this component's resources
    res_indexer = pyo.Set(initialize=range(len(m.resource_index_map[comp])))
    setattr(m, f'{name}_res_index_map', res_indexer)
    prod_name = f'{name}_{tag}'
    init = (((0, t), values[t]) for t in m.T)
    prod = pyo.Param(res_indexer, m.T, initialize=dict(init))
    setattr(m, prod_name, prod)
    return prod_name

  def _create_production(self, m, comp, meta):
    """
      Creates all pyomo variable objects for a non-storage component
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make production variables for
      @ In, meta, dict, dictionary of state variables
      @ Out, None
    """
    prod_name = self._create_production_variable(m, comp, meta)
    ## if you cannot set limits directly in the production variable, set separate contraint:
    ## Method 1: set variable bounds directly --> TODO more work needed, but would be nice
    # lower, upper = self._get_prod_bounds(m, comp)
    # limits should be None unless specified, so use "getters" from dictionaries
    # bounds = lambda m, r, t: (lower.get(r, None), upper.get(r, None))
    ## Method 2: set variable bounds directly --> TODO more work needed, but would be nice
    # self._create_capacity(m, comp, prod_name, meta)    # capacity constraints
    # transfer function governs input -> output relationship
    self._create_transfer(m, comp, prod_name)
    # ramp rates TODO ## INCLUDING previous-time boundary condition TODO

  def _create_production_variable(self, m, comp, meta, tag=None, add_bounds=True, **kwargs):
    """
      Creates production pyomo variable object for a component
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make production variables for
      @ In, tag, str, optional, if not None then name will be component_[tag]; otherwise "production"
      @ In, add_bounds, bool, optional, if True then determine and set bounds for variable
      @ In, kwargs, dict, optional, passalong kwargs to pyomo variable
      @ Out, prod_name, str, name of production variable
    """
    if tag is None:
      tag = 'production'
    name = comp.name
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    limit_r = m.resource_index_map[comp][cap_res] # production index of the governing resource
    # create pyomo indexer for this component's resources
    indexer_name = f'{name}_res_index_map'
    indexer = getattr(m, indexer_name, None)
    if indexer is None:
      indexer = pyo.Set(initialize=range(len(m.resource_index_map[comp])))
      setattr(m, indexer_name, indexer)
    prod_name = f'{name}_{tag}'
    caps, mins = self._find_production_limits(m, comp, meta)
    if min(caps) < 0:
      # quick check that capacities signs are consistent #FIXME: revisit, this is an assumption
      assert max(caps) < 0, \
        'Capacities are inconsistent: mix of positive and negative values not currently  supported.'
      # we have a unit that's consuming, so we need to flip the variables to be sensible
      mins, caps = caps, mins
      inits = caps
    else:
      inits = mins
    if add_bounds:
      # create bounds based in min, max operation
      bounds = lambda m, r, t: (mins[t] if r == limit_r else None, caps[t] if r == limit_r else None)
      initial = lambda m, r, t: inits[t] if r == limit_r else 0
    else:
      bounds = (None, None)
      initial = 0
    # production variable depends on resources, time
    #FIXME initials! Should be lambda with mins for tracking var!
    prod = pyo.Var(indexer, m.T, initialize=initial, bounds=bounds, **kwargs)
    # TODO it may be that we need to set variable values to avoid problems in some solvers.
    # if comp.is_dispatchable() == 'fixed':
    #   for t, _ in enumerate(m.Times):
    #     prod[limit_r, t].fix(caps[t])
    setattr(m, prod_name, prod)
    return prod_name

  def _create_capacity_constraints(self, m, comp, prod_name, meta):
    """
      Creates pyomo capacity constraints
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make variables for
      @ In, prod_name, str, name of production variable
      @ In, meta, dict, additional state information
      @ Out, None
    """
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    r = m.resource_index_map[comp][cap_res] # production index of the governing resource
    caps, mins = self._find_production_limits(m, comp, meta)
    # capacity
    max_rule = lambda mod, t: self._capacity_rule(prod_name, r, caps, mod, t)
    constr = pyo.Constraint(m.T, rule=max_rule)
    setattr(m, '{c}_{r}_capacity_constr'.format(c=comp.name, r=cap_res), constr)
    # minimum
    min_rule = lambda mod, t: self._min_prod_rule(prod_name, r, caps, mins, mod, t)
    constr = pyo.Constraint(m.T, rule=min_rule)
    # set initial conditions
    for t, time in enumerate(m.Times):
      cap = caps[t]
      if cap == mins[t]:
        # initialize values so there's no boundary errors
        var = getattr(m, prod_name)
        values = var.get_values()
        for k in values:
          values[k] = cap
        var.set_values(values)
    setattr(m, '{c}_{r}_minprod_constr'.format(c=comp.name, r=cap_res), constr)

  def _find_production_limits(self, m, comp, meta):
    """
      Determines the capacity limits of a unit's operation, in time.
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make variables for
      @ In, meta, dict, additional state information
      @ Out, caps, array, max production values by time
      @ Out, mins, array, min production values by time
    """
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    r = m.resource_index_map[comp][cap_res] # production index of the governing resource
    # production is always lower than capacity
    ## NOTE get_capacity returns (data, meta) and data is dict
    ## TODO does this work with, e.g., ARMA-based capacities?
    ### -> "time" is stored on "m" and could be used to correctly evaluate the capacity
    caps = []
    mins = []
    for t, time in enumerate(m.Times):
      meta['HERON']['time_index'] = t + m.time_offset
      cap = comp.get_capacity(meta)[0][cap_res] # value of capacity limit (units of governing resource)
      caps.append(cap)
      if (comp.is_dispatchable() == 'fixed'):
        minimum = cap
      else:
        minimum = comp.get_minimum(meta)[0][cap_res]
      mins.append(minimum)
    return caps, mins

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
      rule = lambda mod, t: self._transfer_rule(ratio, r, ref_r, prod_name, mod, t)
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, rule_name, constr)

  def _create_storage(self, m, comp, initial_storage, meta):
    """
      Creates storage pyomo variable objects for a storage component
      Similar to create_production, but for storages
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make production variables for
      @ In, initial_storage, dict, initial storage levels
      @ In, meta, dict, additional state information
      @ Out, level_name, str, name of storage level variable
    """
    prefix = comp.name
    # what resource index? Isn't it always 0? # assumption
    r = 0 # NOTE this is only true if each storage ONLY uses 1 resource
    # storages require a few variables:
    # (1) a level tracker,
    level_name = self._create_production_variable(m, comp, meta, tag='level')
    # -> set operational limits
    # self._create_capacity(m, comp, level_name, meta)
    # (2, 3) separate charge/discharge trackers, so we can implement round-trip efficiency and ramp rates
    charge_name = self._create_production_variable(m, comp, meta, tag='charge', add_bounds=False, within=pyo.NonPositiveReals)
    discharge_name = self._create_production_variable(m, comp, meta, tag='discharge', add_bounds=False, within=pyo.NonNegativeReals)
    # balance level, charge/discharge
    level_rule_name = prefix + '_level_constr'
    rule = lambda mod, t: self._level_rule(comp, level_name, charge_name, discharge_name,
                                           initial_storage, r, mod, t)
    setattr(m, level_rule_name, pyo.Constraint(m.T, rule=rule))
    # (4) a binary variable to track whether we're charging or discharging, to prevent BOTH happening
    # -> 0 is charging, 1 is discharging
    # -> TODO make this a user-based option to disable, if they want to allow dual operation
    # -> -> but they should really think about if that's what they want!
    # FIXME currently introducing the bigM strategy also makes solves numerically unstable,
    # and frequently results in spurious errors. For now, disable it.
    allow_both = True # allow simultaneous charging and discharging
    if not allow_both:
      bin_name = self._create_production_variable(m, comp, meta, tag='dcforcer', add_bounds=False, within=pyo.Binary)
      # we need a large epsilon, but not so large that addition stops making sense
      # -> we don't know what any values for this component will be! How do we choose?
      # -> NOTE that choosing this value has VAST impact on solve stability!!
      large_eps = 1e8 #0.01 * sys.float_info.max
      # charging constraint: don't charge while discharging (note the sign matters)
      charge_rule_name = prefix + '_charge_constr'
      rule = lambda mod, t: self._charge_rule(charge_name, bin_name, large_eps, r, mod, t)
      setattr(m, charge_rule_name, pyo.Constraint(m.T, rule=rule))
      discharge_rule_name = prefix + '_discharge_constr'
      rule = lambda mod, t: self._discharge_rule(discharge_name, bin_name, large_eps, r, mod, t)
      setattr(m, discharge_rule_name, pyo.Constraint(m.T, rule=rule))

  def _create_conservation(self, m, resources, initial_storage, meta):
    """
      Creates pyomo conservation constraints
      @ In, m, pyo.ConcreteModel, associated model
      @ In, resources, list, list of resources in problem
      @ In, initial_storage, dict, initial storage levels
      @ In, meta, dict, dictionary of state variables
      @ Out, None
    """
    for res, resource in enumerate(resources):
      rule = lambda mod, t: self._conservation_rule(initial_storage, meta, resource, mod, t)
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, '{r}_conservation'.format(r=resource), constr)

  def _create_objective(self, meta, m):
    """
      Creates pyomo objective function
      @ In, meta, dict, additional variables to pass through
      @ In, m, pyo.ConcreteModel, associated model
      @ Out, None
    """
    # cashflow eval
    rule = lambda mod: self._cashflow_rule(meta, mod)
    m.obj = pyo.Objective(rule=rule, sense=pyo.maximize)

  ### UTILITIES for general use
  def _get_prod_bounds(self, m, comp, meta):
    """
      Determines the production limits of the given component
      @ In, comp, HERON component, component to get bounds of
      @ In, meta, dict, additional state information
      @ Out, lower, dict, lower production limit (might be negative for consumption)
      @ Out, upper, dict, upper production limit (might be 0 for consumption)
    """
    raise NotImplementedError
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    r = m.resource_index_map[comp][cap_res]
    maxs = []
    mins = []
    for t, time in enumerate(m.Times):
      meta['HERON']['time_index'] = t + m.time_offset
      cap = comp.get_capacity(meta)[0][cap_res]
      low = comp.get_minimum(meta)[0][cap_res]
      maxs.append(cap)
      if (comp.is_dispatchable() == 'fixed') or (low == cap):
        low = cap
        # initialize values to avoid boundary errors
        var = getattr(m, prod_name)
        values = var.get_values()
        for k in values:
          values[k] = cap
        var.set_values(values)
      mins.append(low)
    maximum = comp.get_capacity(None, None, None, None)[0][cap_res]
    # TODO minimum!
    # producing or consuming the defining resource?
    # -> put these in dictionaries so we can "get" limits or return None
    if maximum > 0:
      lower = {r: 0}
      upper = {r: maximum}
    else:
      lower = {r: maximum}
      upper = {r: 0}
    return lower, upper

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
    coeffs = transfer.get_coefficients() # linear transfer coefficients, dict as {resource: coeff}, SIGNS MATTER
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
      @ Out, result, dict, {comp: {activity: {resource: [production]}} e.g. generator[production][steam]
    """
    result = {} # {component: {activity_tag: {resource: production}}}
    for comp in m.Components:
      result[comp.name] = {}
      for tag in comp.get_tracking_vars():
        tag_values = self._retrieve_value_from_model(m, comp, tag)
        result[comp.name][tag] = tag_values
    return result

  def _retrieve_value_from_model(self, m, comp, tag):
    """
      Retrieve values of a series from the pyomo model.
      @ In, m, pyo.ConcreteModel, associated (solved) model
      @ In, comp, Component, relevant component
      @ In, tag, str, particular history type to retrieve
      @ Out, result, dict, {resource: [array], etc}
    """
    result = {}
    prod = getattr(m, f'{comp.name}_{tag}')
    if isinstance(prod, pyo.Var):
      kind = 'Var'
    elif isinstance(prod, pyo.Param):
      kind = 'Param'
    for res, comp_r in m.resource_index_map[comp].items():
      if kind == 'Var':
        result[res] = np.fromiter((prod[comp_r, t].value for t in m.T), dtype=float, count=len(m.T))
      elif kind == 'Param':
        result[res] = np.fromiter((prod[comp_r, t] for t in m.T), dtype=float, count=len(m.T))
    return result

  ### RULES for lambda function calls
  # these get called using lambda functions to make Pyomo constraints, vars, objectives, etc

  def _charge_rule(self, charge_name, bin_name, large_eps, r, m, t):
    """
      Constructs pyomo don't-charge-while-discharging constraints.
      For storage units specificially.
      @ In, charge_name, str, name of charging variable
      @ In, bin_name, str, name of forcing binary variable
      @ In, large_eps, float, a large-ish number w.r.t. storage activity
      @ In, r, int, index of stored resource (is this always 0?)
      @ In, m, pyo.ConcreteModel, associated model
      @ In, t, int, time index for capacity rule
      @ Out, rule, bool, inequality used to limit charge behavior
    """
    charge_var = getattr(m, charge_name)
    bin_var = getattr(m, bin_name)
    return -charge_var[r, t] <= (1 - bin_var[r, t]) * large_eps

  def _discharge_rule(self, discharge_name, bin_name, large_eps, r, m, t):
    """
      Constructs pyomo don't-discharge-while-charging constraints.
      For storage units specificially.
      @ In, discharge_name, str, name of discharging variable
      @ In, bin_name, str, name of forcing binary variable
      @ In, large_eps, float, a large-ish number w.r.t. storage activity
      @ In, r, int, index of stored resource (is this always 0?)
      @ In, m, pyo.ConcreteModel, associated model
      @ In, t, int, time index for capacity rule
      @ Out, rule, bool, inequality used to limit discharge behavior
    """
    discharge_var = getattr(m, discharge_name)
    bin_var = getattr(m, bin_name)
    return discharge_var[r, t] <= bin_var[r, t] * large_eps

  def _level_rule(self, comp, level_name, charge_name, discharge_name, initial_storage, r, m, t):
    """
      Constructs pyomo charge-discharge-level balance constraints.
      For storage units specificially.
      @ In, comp, Component, storage component of interest
      @ In, level_name, str, name of level-tracking variable
      @ In, charge_name, str, name of charging variable
      @ In, discharge_name, str, name of discharging variable
      @ In, r, int, index of stored resource (is this always 0?)
      @ In, m, pyo.ConcreteModel, associated model
      @ In, t, int, time index for capacity rule
      @ Out, rule, bool, inequality used to limit level behavior
    """
    level_var = getattr(m, level_name)
    charge_var = getattr(m, charge_name)
    discharge_var = getattr(m, discharge_name)
    if t > 0:
      previous = level_var[r, t-1]
      dt = m.Times[t] - m.Times[t-1]
    else:
      previous = initial_storage[comp]
      dt = m.Times[1] - m.Times[0]
    rte2 = comp.get_sqrt_RTE() # square root of the round-trip efficiency
    production = - rte2 * charge_var[r, t] - discharge_var[r, t] / rte2
    return level_var[r, t] == previous + production * dt

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
    total = self._compute_cashflows(m.Components, activity, m.Times, meta,
                                    state_args=state_args, time_offset=m.time_offset)
    return total

  def _conservation_rule(self, initial_storage, meta, res, m, t):
    """
      Constructs conservation constraints.
      @ In, initial_storage, dict, initial storage levels at t==0 (not t+offset==0)
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
        r = res_dict[res]
        intr = comp.get_interaction()
        if intr.is_type('Storage'):
          # Storages have 3 variables: level, charge, and discharge
          # -> so calculate activity
          charge = getattr(m, f'{comp.name}_charge')
          discharge = getattr(m, f'{comp.name}_discharge')
          # note that "charge" is negative (as it's consuming) and discharge is positive
          # -> so the intuitive |discharge| - |charge| becomes discharge + charge
          production = discharge[r, t] + charge[r, t]
        else:
          var = getattr(m, f'{comp.name}_production')
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
      for tracker in comp.get_tracking_vars():
        for res, r in m.resource_index_map[comp].items():
          print(f'    tracker: {tracker} resource {r}: {res}')
          for t, time in enumerate(m.Times):
            prod = getattr(m, f'{name}_{tracker}')
            if isinstance(prod, pyo.Var):
              print('      time:', t + m.time_offset, time, prod[r, t].value)
            elif isinstance(prod, pyo.Param):
              print('      time:', t + m.time_offset, time, prod[r, t])
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

  def get_activity_indexed(self, comp, activity, r, t, valued=True, **kwargs):
    """
      Getter for activity level.
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, activity, str, tracking variable name for activity subset
      @ In, r, int, index of resource to retrieve (as given by meta[HERON][resource_indexer])
      @ In, t, int, index of time at which activity should be provided
      @ In, valued, bool, optional, if True then get float value instead of pyomo expression
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    prod = getattr(self._model, f'{comp.name}_{activity}')[r, t]
    if valued:
      return prod()
    return prod

  def set_activity_indexed(self, comp, r, t, value, valued=False):
    raise NotImplementedError

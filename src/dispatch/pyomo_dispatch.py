
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  pyomo-based dispatch strategy
"""
import time as time_mod
import platform
import pprint
import numpy as np
import pyutilib.subprocess.GlobalData
from itertools import compress

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.common.errors import ApplicationError
from ravenframework.utils import InputData, InputTypes

from . import PyomoRuleLibrary as prl
from . import putils
from .Dispatcher import Dispatcher
from .DispatchState import DispatchState, NumpyState, PyomoState
from .. import _utils as hutils


# allows pyomo to solve on threaded processes
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
# Choose solver; CBC is a great choice unless we're on Windows
if platform.system() == 'Windows':
  SOLVERS = ['glpk', 'cbc', 'ipopt']
else:
  SOLVERS = ['cbc', 'glpk', 'ipopt']

# different solvers express "tolerance" for converging solution in different
# ways. Further, they mean different things for different solvers. This map
# just tracks the "nominal" argument that we should pass through pyomo.
solver_tol_map = {
  'ipopt': 'tol',
  'cbc': 'primalTolerance',
  'glpk': 'mipgap',
}

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
    specs.addSub(InputData.parameterInputFactory('tol', contentType=InputTypes.FloatType,
        descr=r"""Relative tolerance for converging final optimal dispatch solutions. Specific implementation
        depends on the solver selected. Changing this value could have significant impacts on the
        dispatch optimization time and quality.
        \default{solver dependent, often 1e-6}."""))
    # TODO specific for pyomo dispatcher
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.name = 'PyomoDispatcher' # identifying name
    self.debug_mode = False       # whether to print additional information
    self.solve_options = {}       # options passed from Pyomo to the solver
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

    # the tolerance value needs to be saved for after the solver is set
    # since we don't know what key to assign it to otherwise
    tol_node = specs.findFirst('tol')
    if tol_node is not None:
      solver_tol = tol_node.value
    else:
      solver_tol = None

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

    if solver_tol is not None:
      key = solver_tol_map[self._solver]
      self.solve_options[key] = solver_tol

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
      print(f'DEBUGG starting window {start_index} to {end_index}')
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
      print(f'DEBUGG solve time: {end-start} s')
      # store result in corresponding part of dispatch
      for comp in components:
        for tag in comp.get_tracking_vars():
          for res, values in subdisp[comp.name][tag].items():
            dispatch.set_activity_vector(comp, res, values, tracker=tag, start_idx=start_index, end_idx=end_index)
      start_index = end_index
    return dispatch

  def get_solver(self):
    """
      Retrieves the solver information (if applicable)
      @ In, None
      @ Out, solver, str, name of solver used
    """
    return self._solver

  ### INTERNAL
  @staticmethod
  def _calculate_deltas(activity, initial_level):
    """
    """
    deltas = np.zeros(len(activity))
    deltas[1:] = activity[1:] - activity[:-1]
    deltas[0] = activity[0] - initial_level
    return deltas

  def _process_storage_component(self, m, comp, intr, meta):
    """
    """
    activity = intr.get_strategy().evaluate(meta)[0]["level"]
    self._create_production_param(m, comp, activity, tag="level")
    dt = m.Times[1] - m.Times[0]
    rte2 = comp.get_sqrt_RTE()
    deltas = self._calculate_deltas(activity, intr.get_initial_level(meta))
    charge = np.where(deltas > 0, -deltas / dt / rte2, 0)
    discharge = np.where(deltas < 0, -deltas / dt * rte2, 0)
    self._create_production_param(m, comp, charge, tag="charge")
    self._create_production_param(m, comp, discharge, tag="discharge")

  def _process_components(self, m, comp, time, initial_storage, meta):
    """
    """
    intr = comp.get_interaction()
    if intr.is_governed():
      self._process_governed_component(m, comp, time, intr, meta)
    elif intr.is_type("Storage"):
      self._create_storage(m, comp, initial_storage, meta)
    else:
      self._create_production(m, comp, meta)

  def _process_governed_component(self, m, comp, time, intr, meta):
    """
    """
    meta["request"] = {"component": comp, "time": time}
    if intr.is_type("Storage"):
      self._process_storage_component(m, comp, intr, meta)
    else:
      activity = intr.get_strategy().evaluate(meta)[0]['level']
      self._create_production_param(m, comp, activity)
      
  def _build_pyomo_model(self, time, time_offset, case, components, resources, meta):
    """
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
    m.time_offset = time_offset
    m.resource_index_map = meta['HERON']['resource_indexer'] # maps the resource to its index WITHIN APPLICABLE components (sparse matrix)
                                                             #   e.g. component: {resource: local index}, ... etc}
    # properties
    m.Case = case
    m.Components = components
    m.Activity = PyomoState()
    m.Activity.initialize(m.Components, m.resource_index_map, m.Times, m)
    return m
  
  def _populate_pyomo_model(self, model, components, initial_storage, time, resources, meta):
    """
    """
    # constraints and variables
    for comp in components:
      self._process_components(model, comp, time, initial_storage, meta)
    self._create_conservation(model, resources, initial_storage, meta) # conservation of resources (e.g. production == consumption)
    self._create_objective(meta, model) # objective

  def dispatch_window(self, time, time_offset, case, components, sources, resources, initial_storage, meta):
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
    model = self._build_pyomo_model(time, time_offset, case, components, resources, meta)
    self._populate_pyomo_model(model, components, initial_storage, time, resources, meta)
    result = self._solve_dispatch(model, meta)
    return result
    
  def _solve_dispatch(self, m, meta):
    """
    """
    # start a solution search
    done_and_checked = False
    attempts = 0
    # DEBUGG show variables, bounds
    if self.debug_mode:
      putils.debug_pyomo_print(m)
    while not done_and_checked:
      attempts += 1
      print(f'DEBUGG solve attempt {attempts} ...:')
      # solve
      soln = pyo.SolverFactory(self._solver).solve(m, options=self.solve_options)
      # check solve status
      if soln.solver.status == SolverStatus.ok and soln.solver.termination_condition == TerminationCondition.optimal:
        print('DEBUGG ... solve was successful!')
      else:
        print('DEBUGG ... solve was unsuccessful!')
        print('DEBUGG ... status:', soln.solver.status)
        print('DEBUGG ... termination:', soln.solver.termination_condition)
        putils.debug_pyomo_print(m)
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
          print(f"DEBUGG ... ... Time {e['time_index']} ({e['time']}) \n" +
                f"Component \"{e['component'].name}\" Resource \"{e['resource']}\": {e['msg']}")
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
      putils.debug_print_soln(m)
    # return dict of numpy arrays
    result = putils.retrieve_solution(m)
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
    rule = lambda mod: prl.prod_limit_rule(prod_name, r, limits, limit_type, t, mod)
    constr = pyo.Constraint(rule=rule)
    counter = 1
    name_template = f'{comp.name}_{resource}_{t}_vld_limit_constr_{{i}}'
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
    # ramp rates
    if comp.ramp_limit is not None:
      self._create_ramp_limit(m, comp, prod_name, meta)
    return prod_name

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
      assert max(caps) <= 0, \
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

  def _create_ramp_limit(self, m, comp, prod_name, meta):
    """
      Creates ramping limitations for a producing component
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make ramping limits for
      @ In, prod_name, str, name of production variable
      @ In, meta, dict, dictionary of state variables
      @ Out, None
    """
    # ramping is defined in terms of the capacity variable
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    cap = comp.get_capacity(meta)[0][cap_res]
    r = m.resource_index_map[comp][cap_res] # production index of the governing resource
    # NOTE: this includes the built capacity * capacity factor, if any, which assumes
    # the ramp rate depends on the available capacity, not the built capacity.
    limit_delta = comp.ramp_limit * cap # NOTE: if cap is negative, then this is negative.
    if limit_delta < 0:
      neg_cap = True
    else:
      neg_cap = False
    # if we're limiting ramp frequency, make vars and rules for that
    if comp.ramp_freq:
      # create binaries for tracking ramping
      up = pyo.Var(m.T, initialize=0, domain=pyo.Binary)
      down = pyo.Var(m.T, initialize=0, domain=pyo.Binary)
      steady = pyo.Var(m.T, initialize=1, domain=pyo.Binary)
      setattr(m, f'{comp.name}_up_ramp_tracker', up)
      setattr(m, f'{comp.name}_down_ramp_tracker', down)
      setattr(m, f'{comp.name}_steady_ramp_tracker', steady)
      ramp_trackers = (down, up, steady)
    else:
      ramp_trackers = None
    # limit production changes when ramping down
    ramp_rule_down = lambda mod, t: prl.ramp_rule_down(prod_name, r, limit_delta, neg_cap, t, mod, bins=ramp_trackers)
    constr = pyo.Constraint(m.T, rule=ramp_rule_down)
    setattr(m, f'{comp.name}_ramp_down_constr', constr)
    # limit production changes when ramping up
    ramp_rule_up = lambda mod, t: prl.ramp_rule_up(prod_name, r, limit_delta, neg_cap, t, mod, bins=ramp_trackers)
    constr = pyo.Constraint(m.T, rule=ramp_rule_up)
    setattr(m, f'{comp.name}_ramp_up_constr', constr)
    # if ramping frequency limit, impose binary constraints
    if comp.ramp_freq:
      # binaries rule, for exclusive choice up/down/steady
      binaries_rule = lambda mod, t: prl.ramp_freq_bins_rule(down, up, steady, t, mod)
      constr = pyo.Constraint(m.T, rule=binaries_rule)
      setattr(m, f'{comp.name}_ramp_freq_binaries', constr)
      # limit frequency of ramping
      # TODO calculate "tao" window using ramp freq and dt
      # -> for now, just use the integer for number of windows
      freq_rule = lambda mod, t: prl.ramp_freq_rule(down, up, comp.ramp_freq, t, m)
      constr = pyo.Constraint(m.T, rule=freq_rule)
      setattr(m, f'{comp.name}_ramp_freq_constr', constr)

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
    max_rule = lambda mod, t: prl.capacity_rule(prod_name, r, caps, mod, t)
    constr = pyo.Constraint(m.T, rule=max_rule)
    setattr(m, f'{comp.name}_{cap_res}_capacity_constr', constr)
    # minimum
    min_rule = lambda mod, t: prl.min_prod_rule(prod_name, r, caps, mins, mod, t)
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
    setattr(m, f'{comp.name}_{cap_res}_minprod_constr', constr)

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
    ratios = putils.get_transfer_coeffs(m, comp)
    ref_r, ref_name, _ = ratios.pop('__reference', (None, None, None))
    for resource, ratio in ratios.items():
      r = m.resource_index_map[comp][resource]
      rule_name = f'{name}_{resource}_{ref_name}_transfer'
      rule = lambda mod, t: prl.transfer_rule(ratio, r, ref_r, prod_name, mod, t)
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
    rule = lambda mod, t: prl.level_rule(comp, level_name, charge_name, discharge_name, initial_storage, r, mod, t)
    setattr(m, level_rule_name, pyo.Constraint(m.T, rule=rule))
    # periodic boundary condition for storage level
    if comp.get_interaction().apply_periodic_level:
      periodic_rule_name = prefix + '_level_periodic_constr'
      rule = lambda mod, t: prl.periodic_level_rule(comp, level_name, initial_storage, r, mod, t)
      setattr(m, periodic_rule_name, pyo.Constraint(m.T, rule=rule))

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
      rule = lambda mod, t: prl.charge_rule(charge_name, bin_name, large_eps, r, mod, t)
      setattr(m, charge_rule_name, pyo.Constraint(m.T, rule=rule))
      discharge_rule_name = prefix + '_discharge_constr'
      rule = lambda mod, t: prl.discharge_rule(discharge_name, bin_name, large_eps, r, mod, t)
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
    for resource in resources:
      rule = lambda mod, t: prl.conservation_rule(resource, mod, t)
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, f'{resource}_conservation', constr)

  def _create_objective(self, meta, m):
    """
      Creates pyomo objective function
      @ In, meta, dict, additional variables to pass through
      @ In, m, pyo.ConcreteModel, associated model
      @ Out, None
    """
    # cashflow eval
    rule = lambda mod: prl.cashflow_rule(self._compute_cashflows, meta, mod)
    m.obj = pyo.Objective(rule=rule, sense=pyo.maximize)

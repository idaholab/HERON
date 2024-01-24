# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  pyomo-based dispatch strategy
"""
import time as time_mod
import pprint
import numpy as np
import pyutilib.subprocess.GlobalData

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from ravenframework.utils import InputData, InputTypes

from . import putils
from .PyomoModelHandler import PyomoModelHandler
from .Dispatcher import Dispatcher, DispatchError
from .DispatchState import NumpyState

# allows pyomo to solve on threaded processes
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

# different solvers express "tolerance" for converging solution in different
# ways. Further, they mean different things for different solvers. This map
# just tracks the "nominal" argument that we should pass through pyomo.
SOLVER_TOL_MAP = {
  'ipopt': 'tol',
  'cbc': 'primalTolerance',
  'glpk': 'mipgap',
}

class Pyomo(Dispatcher):
  """
    Dispatches using rolling windows in Pyomo
  """
  # naming_template = {
  #   'comp prod': '{comp}|{res}|prod',
  #   'comp transfer': '{comp}|{res}|trans',
  #   'comp max': '{comp}|{res}|max',
  #   'comp ramp up': '{comp}|{res}|rampup',
  #   'comp ramp down': '{comp}|{res}|rampdown',
  #   'conservation': '{res}|consv',
  # }

  ### INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory(
      'pyomo', ordered=False, baseNode=None,
      descr=r"""The \texttt{pyomo} dispatcher uses analytic modeling and rolling
      windows to solve dispatch optimization with perfect information via the
      pyomo optimization library."""
    )

    specs.addSub(
      InputData.parameterInputFactory(
        'rolling_window_length', contentType=InputTypes.IntegerType,
        descr=r"""Sets the length of the rolling window that the Pyomo optimization
        algorithm uses to break down histories. Longer window lengths will minimize
        boundary effects, such as nonoptimal storage dispatch, at the cost of slower
        optimization solves. Note that if the rolling window results in a window
        of length 1 (such as at the end of a history), this can cause problems for pyomo.
        \default{24}"""
      )
    )

    specs.addSub(
      InputData.parameterInputFactory(
        'debug_mode', contentType=InputTypes.BoolType,
        descr=r"""Enables additional printing in the pyomo dispatcher.
        Highly discouraged for production runs. \default{False}."""
      )
    )

    specs.addSub(
      InputData.parameterInputFactory(
        'solver', contentType=InputTypes.StringType,
        descr=r"""Indicates which solver should be used by pyomo. Options depend
        on individual installation. \default{'glpk' for Windows, 'cbc' otherwise}."""
      )
    )

    specs.addSub(
      InputData.parameterInputFactory(
        'tol', contentType=InputTypes.FloatType,
        descr=r"""Relative tolerance for converging final optimal dispatch solutions.
        Specific implementation depends on the solver selected. Changing this value
        could have significant impacts on the dispatch optimization time and quality.
        \default{solver dependent, often 1e-6}."""
      )
    )
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


  def read_input(self, specs) -> None:
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

    tol_node = specs.findFirst('tol')
    if tol_node is not None:
      solver_tol = tol_node.value
    else:
      solver_tol = None

    self._solver = putils.check_solver_availability(self._solver)

    if solver_tol is not None:
      key = SOLVER_TOL_MAP.get(self._solver, None)
      if key is not None:
        self.solve_options[key] = solver_tol
      else:
        raise ValueError(f"Tolerance setting not available for solver '{self._solver}'.")


  def get_solver(self):
    """
      Retrieves the solver information (if applicable)
      @ In, None
      @ Out, solver, str, name of solver used
    """
    return self._solver


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
    time = np.linspace(t_start, t_end, t_num)
    resources = sorted(putils.get_all_resources(components))
    dispatch = NumpyState()
    dispatch.initialize(components, meta['HERON']['resource_indexer'], time)

    start_index = 0
    final_index = len(time)
    initial_levels = putils.get_initial_storage_levels(components, meta, start_index)

    while start_index < final_index:
      end_index = min(start_index + self._window_len, final_index)
      if end_index - start_index == 1:
        raise DispatchError("Window length of 1 detected, which is not supported.")

      specific_time = time[start_index:end_index]
      print(f"Start: {start_index} End: {end_index}")
      subdisp, solve_time = self._handle_dispatch_window_solve(
        specific_time, start_index, case, components, sources, resources, initial_levels, meta
      )
      print(f'DEBUGG solve time: {solve_time} s')

      # Store Results of optimization into dispatch container
      for comp in components:
        for tag in comp.get_tracking_vars():
          for res, values in subdisp[comp.name][tag].items():
            dispatch.set_activity_vector(comp, res, values, tracker=tag, start_idx=start_index, end_idx=end_index)
      start_index = end_index

    return dispatch


  def _handle_dispatch_window_solve(self, specific_time, start_index, case, components, sources, resources, initial_levels, meta):
    """
      Set up convergence criteria and collect results from a dispatch window solve.
      @ In, specific_time, np.array, value of time to evaluate.
      @ In, start_index, int, index of the start of the window.
      @ In, case, HERON Case, Case that this dispatch is part of.
      @ In, components, list, HERON components available to the dispatch.
      @ In, sources, list, HERON source (placeholders) for signals.
      @ In, resources, list, sorted list of all resources in problem.
      @ In, initial_levels, dict, initial storage levels if any.
      @ In, meta, dict, additional variables passed through.
      @ Out, subdisp, dict, results of window dispatch.
    """
    start = time_mod.time()
    subdisp = self._dispatch_window(specific_time, start_index, case, components, resources, initial_levels, meta)

    if self._needs_convergence(components):
      conv_counter = 0
      converged = False
      previous = None

      while not converged and conv_counter < self._picard_limit:
        conv_counter += 1
        print(f'DEBUGG iteratively solving window, iteration {conv_counter}/{self._picard_limit} ...')
        subdisp = self._dispatch_window(specific_time, start_index, case, components, resources, initial_levels, meta)
        converged = self._check_if_converged(subdisp, previous, components)
        previous = subdisp

      if conv_counter >= self._picard_limit and not converged:
        raise DispatchError(f"Convergence not reached after {self._picard_limit} iterations.")

    end = time_mod.time()
    solve_time = end - start
    return subdisp, solve_time


  def _dispatch_window(self, time, time_offset, case, components, resources, initial_storage, meta):
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
    model = PyomoModelHandler(time, time_offset, case, components, resources, initial_storage, meta)
    model.populate_model()
    result = self._solve_dispatch(model, meta)
    return result


  def _check_if_converged(self, new, old, components, tol=1e-4):
    """
      Checks convergence of consecutive dispatch solves
      @ In, new, dict, results of dispatch # TODO should this be the model rather than dict?
      @ In, old, dict, results of previous dispatch
      @ In, components, list, HERON component list
      @ In, tol, float, optional, tolerance for convergence
      @ Out, converged, bool, True if convergence is met
    """
    if old is None:
      return False

    for comp in components:
      intr = comp.get_interaction()
      if intr.is_governed(): # by "is_governed" we mean "isn't optimized in pyomo"
        # check activity L2 norm as a differ
        # TODO this may be specific to storage right now
        name = comp.name
        tracker = comp.get_tracking_vars()[0]
        res = intr.get_resource()
        scale = np.max(old[name][tracker][res])
        # Avoid division by zero
        if scale == 0:
          diff = np.linalg.norm(new[name][tracker][res] - old[name][tracker][res])
        else:
          diff = np.linalg.norm(new[name][tracker][res] - old[name][tracker][res]) / scale
        if diff > tol:
          return False
    return True


  def _needs_convergence(self, components):
    """
      Determines whether the current setup needs convergence to solve.
      @ In, components, list, HERON component list
      @ Out, needs_convergence, bool, True if iteration is needed
    """
    return any(comp.get_interaction().is_governed() for comp in components)


  def _solve_dispatch(self, m, meta):
    """
      Solves the dispatch problem.
      @ In, m, PyomoModelHandler, model object to solve
      @ In, meta, dict, additional variables passed through
      @ Out, result, dict, results of solved dispatch
    """
    # start a solution search
    done_and_checked = False
    attempts = 0
    # DEBUGG show variables, bounds
    if self.debug_mode:
      putils.debug_pyomo_print(m.model)

    while not done_and_checked:
      attempts += 1
      print(f'DEBUGG using solver: {self._solver}')
      print(f'DEBUGG solve attempt {attempts} ...:')
      soln = pyo.SolverFactory(self._solver).solve(m.model, options=self.solve_options)

      # check solve status
      if soln.solver.status == SolverStatus.ok and soln.solver.termination_condition == TerminationCondition.optimal:
        print('DEBUGG ... solve was successful!')
      else:
        putils.debug_pyomo_print(m.model)
        print('Resource Map:')
        pprint.pprint(m.model.resource_index_map)
        raise DispatchError(
          f"Solve was unsuccessful! Status: {soln.solver.status} Termination: {soln.solver.termination_condition}"
        )

      # try validating
      print('DEBUGG ... validating ...')
      validation_errs = self.validate(m.model.Components, m.model.Activity, m.model.Times, meta)
      if validation_errs:
        done_and_checked = False
        print('DEBUGG ... validation concerns raised:')
        for e in validation_errs:
          print(f"DEBUGG ... ... Time {e['time_index']} ({e['time']}) \n" +
                f"Component \"{e['component'].name}\" Resource \"{e['resource']}\": {e['msg']}")
          m._create_production_limit(e)
        # TODO go back and solve again
        # raise DispatchError('Validation failed, but idk how to handle that yet')
      else:
        print('DEBUGG Solve successful and no validation concerns raised.')
        done_and_checked = True

      # In the words of Charles Bukowski, "Don't Try [too many times]"
      if attempts > 100:
        raise DispatchError('Exceeded validation attempt limit!')

    if self.debug_mode:
      soln.write()
      putils.debug_print_soln(m.model)

    # return dict of numpy arrays
    result = putils.retrieve_solution(m.model)
    return result

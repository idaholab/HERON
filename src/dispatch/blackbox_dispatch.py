import sys
import os
from typing import List, Callable
from itertools import chain
from pprint import pformat

import numpy as np

from .Dispatcher import Dispatcher
from .DispatchState import NumpyState
from ravenframework.utils import InputData



class BlackboxSolution(object):
  time = None
  dispatch = None
  storage = None
  error = False

  def __init__(self, time, dispatch, storage, error, objval, time_windows=None):
    '''
      Representation of the problem solution.
      @ In, time, time horizon used in the problem
      @ In, dispatch, an object/dict describing the optimal dispatch of the system
      @ In, storage, an object/dict describing the usage of storage over the time horizon
      @ In, error, the total constraint error of the final solution
      @ In, objval, the final value of the objective function
      @ In, time_windows, description of where the involved windows start and end
    '''
    self.time = time
    self.dispatch = dispatch
    self.storage = storage
    self.error = error
    self.objval = objval
    self.time_windows = time_windows

class BlackboxComponent(object):
  def __init__(self, name: str, capacity: np.ndarray, ramp_rate_up: np.ndarray, ramp_rate_down: np.ndarray,
              capacity_resource: str, transfer: Callable, cost_function: Callable,
              produces=None, consumes=None, stores=None, min_capacity=None, dispatch_type: str='independent',
              guess: np.ndarray=None, storage_init_level=0.0):
    """
      A Component compatible with the PyOptSparse dispatcher
      @ In, name, Name of the component. Used in representing dispatches
      @ In, capacity, Maximum capacity of the component in terms of `capacity_resource`
      @ In, ramp_rate_up, the maximum positive ramp rate of the component in terms of capacity resource units per time
      @ In, ramp_rate_down, the maximum negative ramp rate of the component in terms of capacity resource units per time
      @ In, transfer, a method for calculating the component transfer at a time point
      @ In, cost_function, an function describing the economic cost of running the unit over a given dispatch
      @ In, produces, a list of resources produced by the component
      @ In, consumes, a list of resources consumed by the component
      @ In, stores, resource stored by the component
      @ In, min_capacity, Minimum capacity of the unit at each time point. Defaults to 0.
      @ In, dispatch_type, the dispatch type of the component
      @ In, guess, a guess at the optimal dispatch of the unit in terms of its `capacity_resource`. Defaults to the capacity.
      @ In, storage_init_level, initial storage level
      @ Out, None

      Note that a component cannot store resources in addition to producing or consuming
      resources. Storage components must be separate from producers and consumers.

      Components can only store a single resource at a time. This could be extended in the
      future to interrelated storage of multiple resources.
    """
    self.name = name
    self.capacity = capacity
    self.capacity_resource = capacity_resource
    self.transfer = transfer
    if type(produces) == list:
      self.produces = produces
    else:
      self.produces = [produces]
    # if type(stores) == list:
    #     self.stores = stores
    # else:
    #     self.stores = [stores]
    self.stores = stores
    if type(consumes) == list:
      self.consumes = consumes
    else:
      self.consumes = [consumes]
    self.dispatch_type = dispatch_type

    # Check to make sure that it does interact with at least one resource
    if produces is None and consumes is None and stores is None:
      raise RuntimeWarning('This component does not interact with any resource!')

    should_be_arrays = {
      'capacity': capacity,
      'ramp_rate_up':ramp_rate_up,
      'ramp_rate_down':ramp_rate_down,
    }
    for name, value in should_be_arrays.items():
      if type(value) is not np.ndarray:
        raise TypeError(f'PyOptSparseComponent {name} must be a numpy array')


    if guess is None:
      self.guess = self.capacity
    else:
      if type(guess) is not np.ndarray:
        raise TypeError('PyOptSparseComponent guess must be a numpy array')
      self.guess = guess

    if min_capacity is None:
      self.min_capacity = np.zeros(len(self.capacity))
    else:
      # Check the datatype
      if type(min_capacity) is not np.ndarray:
        raise TypeError('min_capacity must be a numpy array')
      self.min_capacity = min_capacity

    self.ramp_rate_up = ramp_rate_up
    self.ramp_rate_down = ramp_rate_down
    self.cost_function = cost_function
    self.storage_init_level = storage_init_level

  def get_resources(self):
    """
    
    """
    return [t for t in set([*self.produces, self.stores, *self.consumes]) if t]


class BlackboxDispatchState(object):
  '''Modeled after idaholab/HERON NumpyState object'''
  def __init__(self, components: List[BlackboxComponent], time: List[float]):
    """
    @ In, components, a list of BlackboxComponents
    @ In, time, a list of floats
    @ Out, None
    """
    s = {}

    for c in components:
      s[c.name] = {}
      for resource in c.get_resources():
        s[c.name][resource] = np.zeros(len(time))

    self.state = s
    self.time = time

  def set_activity(self, component: BlackboxComponent, resource, activity, i=None):
    """
    @ In, component, a BlackboxComponent
    @ In, resource,
    @ In, activity,
    @ In, i, an integer to use as an index
    @ Out, None
    """
    if i is None:
      self.state[component.name][resource] = activity
    else:
      self.state[component.name][resource][i] = activity

  def get_activity(self, component: BlackboxComponent, resource, i=None):
    """
    @ In, component, a BlackboxComponent
    @ In, resource,
    @ In, i, an integer to use as an index
    @ Out, the state of that resource for that component
    """
    try:
      if i is None:
        return self.state[component.name][resource]
      else:
        return self.state[component.name][resource][i]
    except Exception as err:
      print(i)
      raise err

  def set_activity_vector(self, component: BlackboxComponent,
                          resource, start, end, activity):
    """
    @ In, component, a BlackboxComponent
    @ In, resource,
    @ In, start,
    @ In, end,
    @ In, activity,
    @ Out, None
    """
    self.state[component.name][resource][start:end] = activity

  def __repr__(self):
    """
    @ In, None
    @ Out, None
    """
    return pformat(self.state)


class ChickadeeDispatcher(object):
  '''
  Dispatch using pyOptSparse optimization package and a pool-based method.
  '''

  slack_storage_added = False # In case slack storage is added in a loop

  def __init__(self, window_length=10):
    """
    @ In, window_length, the length of window that you want to evaluate
    @ Out, None
    """
    self.name = 'CyIpopt'
    self._window_length = window_length

    # Defined on call to self.dispatch
    self.components = None
    self.case = None
    self.storage_levels = {}

  def _gen_pool_cons(self, resource, time_array, start, end, init_store) -> callable:
    """
    A closure for generating a pool constraint for a resource
    @ In, resource the resource to evaluate
    @ In, time_array
    @ In, start, start of time window
    @ In, end, end of time window
    @ In, init_store, insitial sotrage level of resource
    @ Out, a function representing the pool constraint
    """

    def pool_cons(x: List[float]) -> float:
      """
      A resource pool constraint
      Checks that the net amount of a resource being consumed, produced and
      stored is zero.
      @ In, x, a list of floats constaining the guess dispatch to evaluate
      @ Out, float, SSE of resource constraint violations
      """
      x_dict = {}
      for i, c in enumerate(self.components):
        if c.dispatch_type != 'fixed':
          x_dict[c.name] = np.array(x[i*len(time_array):(i+1)*len(time_array)])
      dispatch_window, _ = self.determine_dispatch(x_dict, time_array, start, end, init_store)

      err = np.zeros(len(time_array))
      cs = [c for c in self.components if resource in c.get_resources()]
      for i, _ in enumerate(time_array):
        for c in cs:
          if c.stores:
            err[i] += -dispatch_window.get_activity(c, resource, i)
          else:
            err[i] += dispatch_window.get_activity(c, resource, i)

      return -sum(err**2) # This gives it a little slack to get around numerical difficulties

    return pool_cons

  def _gen_pool_cons2(self, resource, time_array, start, end, init_store, t) -> callable:
    """
    A closure for generating a pool constraint for a resource
    @ In, resource, the resource to evaluate
    @ In, time_array,
    @ In, start, the start of the time window
    @ In, end, the end of the time window
    @ In, int_store, the insitial storage level for the resource at the start of the window
    @ In, t,
    @ Out, a function representing the pool constraint
    """

    def pool_cons(x: List[float]) -> float:
      """
      A resource pool constraint
      Checks that the net amount of a resource being consumed, produced and
      stored is zero.
      @ In, x, a list of floats containing the guess dispatch to evaluate
      @ Out, the SSE of resource constraint violations
      """
      x_dict = {}
      i = 0
      for c in self.components:
        if c.dispatch_type != 'fixed':
          x_dict[c.name] = np.array(x[i*len(time_array):(i+1)*len(time_array)])
          i += 1
      dispatch_window, _ = self.determine_dispatch(x_dict, time_array, start, end, init_store)

      err = 0.0
      cs = [c for c in self.components if resource in c.get_resources()]
      for c in cs:
        if c.stores:
          err += -dispatch_window.get_activity(c, resource, t)
        else:
          err += dispatch_window.get_activity(c, resource, t)

      return -(err**2) + 3 # This gives it a little slack

    return pool_cons

  def _build_pool_cons_individual(self, time, start, end, init_store) -> List[callable]:
    """
    Build the pool constraints
    @ In, time,
    @ In, start, the start of the time window
    @ In, end, the end of the time window
    @ In, init_store, the initial storage level at the start of the window
    @ Out, cons, a callable list of pool constraints, one for each resource
    """

    cons = []
    for res in self.resources:
      for t, _ in enumerate(time):
        cons.append(self._gen_pool_cons2(res, time, start, end, init_store, t))
        # Generate the pool constraint here
        # pool_cons = self._gen_pool_cons(res, time, start, end, init_store)
        # cons.extend(pool_cons)
    return cons

  def _build_pool_cons(self, time, start, end, init_store) -> List[callable]:
    """
    Build the pool constraints
    @ In, time
    @ In, start, the start of the time window
    @ In, end, the end of the time window
    @ In, init_store, the initial storage level at the start of the window
    @ Out, cons, a callable list of pool constraints, one for each resource
    """

    cons = []
    for res in self.resources:
      # for t, _ in enumerate(time):
          # cons.append(self._gen_pool_cons(res, time, start, end, init_store, t))
      # Generate the pool constraint here
      pool_cons = self._gen_pool_cons(res, time, start, end, init_store)
      cons.append(pool_cons)
    return cons

  def _gen_ramp_constraint(self, x_index: int, ramp_rate: float, side='upper') -> callable:
    """
    A closure that returns the function that defines the constraint
    @ In, x_index, integer
    @ In, ramp_rate, float
    @ In, side
    """
    def constraint(x: List[float]) -> float:
      """
      @ In, x, a list constaining floats
      @ Out, constraint, float, if constraint is returned as negative then the constraint was met
      """
      # non-negative result means constraint met
      if side == 'upper':
        return ramp_rate - (x[x_index+1] - x[x_index])
      else:
        return (x[x_index+1] - x[x_index]) - ramp_rate
    return constraint

  def _gen_ramp_constraint_between_windows(self, x_index: int, ramp_rate: float, prev_val: float, side='upper'):
    """
    A closure that returns the functions that defines the constraint between windows
    @ In, x_index, integer
    @ In ramp_rate, float, the maximum ramp rate
    @ In, prev_val, float, the value from the last time window
    @ In, side, string, whether it is the upper or lower limit 
    """
    def constraint(x: List[float]) -> float:
      """
      @ In, x, a list of float
      @ Out, constraint, float, if constraint is returned as negative then the constraint was met
      """
      if side == 'upper':
        return ramp_rate - (x[x_index] - prev_val)
      else:
        return (x[x_index] - prev_val) - ramp_rate
    return constraint

  def _gen_ramp_constraints(self, comp, array_indexer, window_length, start_i, prev_win_end):
    """
    Generates the constraints for the ramp rate
    @ In, comp
    @ In, array_indexer
    @ In, window_length
    @ In, start_i, integer, the starting index
    @ In, prev_win_end, the ending index of the last window
    @ Out, constraints 
    """
    # Add the ramping constraints for each component
    # This could be accelerated by implementing analytic jacobians
    # It would also likely be slightly faster if separate closures were used for upper and lower bounds
    print('generating constraints for ', comp.name)
    constraints = []
    # The first window does not have a previous window, so no need to constrain the ramping between windows
    if start_i != 0:
      x_index = array_indexer*window_length
      print('prev-', x_index)
      ramp_rate_max = comp.ramp_rate_up[start_i]
      ramp_up_constraint = self._gen_ramp_constraint_between_windows(x_index, ramp_rate_max, prev_win_end[comp.name], 'upper')
      constraints.append({
          'type': 'ineq',
          'fun': ramp_up_constraint})
      ramp_rate_min = comp.ramp_rate_down[start_i]
      ramp_down_constraint = self._gen_ramp_constraint_between_windows(x_index, ramp_rate_min, prev_win_end[comp.name], 'lower')
      constraints.append({
              'type': 'ineq',
              'fun': ramp_down_constraint})

    for t in range(window_length-1):
      x_index = array_indexer*window_length+t
      print(x_index, x_index+1)
      ramp_rate_max = comp.ramp_rate_up[start_i+t]
      ramp_up_constraint = self._gen_ramp_constraint(x_index, ramp_rate_max, 'upper')
      constraints.append({
          'type': 'ineq',
          'fun': ramp_up_constraint})
      ramp_rate_min = comp.ramp_rate_down[start_i+t]
      ramp_down_constraint = self._gen_ramp_constraint(x_index, ramp_rate_min, 'lower')
      constraints.append({
              'type': 'ineq',
              'fun': ramp_down_constraint})
    return constraints

  def determine_dispatch(self, opt_vars: dict, time: List[float],
                        start_i: int, end_i: int, init_store: dict) -> BlackboxDispatchState:
    """
    Determine the dispatch from a given set of optimization
    vars by running the transfer functions. Returns a Numpy dispatch
    object
    @ In, opt_vars, dict, holder for all the optimization variables
    @ In, time, list, time horizon to dispatch over
    @ In, start_i, int
    @ In, end_i, int
    @ In, init_store, dict
    @ Out, dispatch, BlackboxDispatchState, dispatch of the system
    @ Out, store_levels, dict, storage levels of each storage component over time
    """
    # Initialize the dispatch
    dispatch = BlackboxDispatchState(self.components, time)
    store_lvls = {}
    # Dispatch the fixed components
    fixed_comps = [c for c in self.components if c.dispatch_type == 'fixed']
    for f in fixed_comps:
      dispatch.set_activity(f, f.capacity_resource,
                            f.capacity[start_i:end_i])
    # Dispatch the independent and dependent components using the vars
    disp_comps = [
      c for c in self.components if c.dispatch_type != 'fixed']
    for d in disp_comps:
      dispatch.set_activity(d, d.capacity_resource, opt_vars[d.name])
      if d.stores:
        store_lvls[d.name] = d.transfer(
            opt_vars[d.name], init_store[d.name])
      else:
        bal = d.transfer(opt_vars[d.name])
        for res, values in bal.items():
          dispatch.set_activity(d, res, values)
    return dispatch, store_lvls

  def _dispatch_pool(self) -> BlackboxSolution:
    """
    Dispatch the given system using a resource-pool method
    @ Out, solution, tuple(DispatchState,dict), optimal dispatch of the system and storage levels of the storage components

      Steps:
        - Assemble all the vars into a vars dict
          A set of vars for each dispatchable component including storage elements
          include bound constraints
        - For each time window
            1) Build the pool constraint functions
                Should have one constraint function for each pool
                Each constraint will be a function of all the vars
            2) Set up the objective function as the double integral of the incremental dispatch
            3) Formulate the problem for pyOptSparse
                a) Declare the variables
                b) Declare the constraints
                c) Declare the objective function
                d) set the optimization configuration (IPOPT/SNOPT, CS/FD...)
            4) Run the optimization and handle failed/unfeasible runs
            5) Set the activities on each of the components and return the result
    """

    objval = 0.0

    # Step 1) Find the vars: 1 for each component input where dispatch is not fixed
    vs = {}  # Min/Max tuples of the various input
    for c in self.components:
      if c.dispatch_type == 'fixed':
        # Fixed dispatch components do not contribute to the variables
        continue
      else:  # Independent and dependent dispatch
        lower = c.min_capacity
        upper = c.capacity
        # Note: This assumes everything based off the first point
        if lower[0] < upper[0]:
          vs[c.name] = [lower, upper]
        else:
          vs[c.name] = [upper, lower]
      if c.stores:
        self.storage_levels[c.name] = np.zeros(len(self.time))

    full_dispatch = BlackboxDispatchState(self.components, self.time)

    win_start_i = 0
    win_i = 0
    prev_win_end_i = 0
    prev_win_end = {} # a dict for tracking the final values of dispatchable components in a time window

    time_windows = []

    while win_start_i < len(self.time):
      win_end_i = win_start_i + self._window_length
      if win_end_i > len(self.time):
        win_end_i = len(self.time)

      # If the end time has not changed, then exit
      if win_end_i == prev_win_end_i:
        break

      if self.verbose:
        print(f'win: {win_i}, start: {win_start_i}, end: {win_end_i}')
      time_windows.append([win_start_i, win_end_i])

      win_horizon = self.time[win_start_i:win_end_i]
      if self.verbose:
        print('Dispatching window', win_i)

      # Assemble the "initial storage levels" for the window
      init_store = {}
      storers = [comp for comp in self.components if comp.stores]
      for storer in storers:
        if win_start_i == 0:
          init_store[storer.name] = storer.storage_init_level
        else:
          init_store[storer.name] = self.storage_levels[storer.name][win_start_i-1]

      if win_i == 0:
        win_dispatch, store_lvls, win_obj_val = self._dispatch_window(
            win_horizon, win_start_i, win_end_i, init_store)
      else:
        win_dispatch, store_lvls, win_obj_val = self._dispatch_window(
            win_horizon, win_start_i, win_end_i, init_store, prev_win_end)
      if self.verbose:
        print(f'Optimal dispatch for win {win_i}:', win_dispatch)

      for comp in self.components:
        for res in comp.get_resources():
          full_dispatch.set_activity_vector(
              comp, res, win_start_i, win_end_i,
              win_dispatch.get_activity(comp, res)
          )
        if comp.dispatch_type != 'fixed':
          prev_win_end[comp.name] = win_dispatch.get_activity(
              comp, comp.capacity_resource, -1
          )

        # Update the storage_levels dict
        if comp.stores:
          self.storage_levels[comp.name][win_start_i:win_end_i] = store_lvls[comp.name]

      # Increment the window indexes
      prev_win_end_i = win_end_i
      win_i += 1
      objval += win_obj_val

      # This results in time windows that match up, but do not overlap
      win_start_i = win_end_i

    # FIXME: Return the total error
    solution = BlackboxSolution(self.time, full_dispatch.state, self.storage_levels,
                            False, objval, time_windows=time_windows)
    return solution

  def generate_objective(self) -> callable:
    '''
    Closure that assembles an objective function to minimize the system cost
    @ Out, objective function
    '''
    if self.external_obj_func:
      return self.external_obj_func
    else:

      def objective(dispatch: BlackboxDispatchState) -> float:
        '''
        The objective function. It is broken out to allow for easier scaling.
        @ In, dispatch, the full dispatch of the system
        @ Out, obj,  float, value of the objective function
        '''
        obj = 0.0
        for c in self.components:
          obj += c.cost_function(dispatch.state[c.name])
        return obj
      return objective

  def _dispatch_window(self, time_window: List[float], start_i: int,
                      end_i: int, init_store, prev_win_end: dict=None) -> BlackboxSolution:
    """
    Dispatch a time-window using a resource-pool method
    @ In, time_window, list, The time window to dispatch the system over
    @ In, start_i, int, The time-array index for the start of the window
    @ In, end_i, int, The time-array index for the end of the window
    @ In, init_store, dict, the initial storage values of the storage components
    @ In, prev_win_end, dict, the ending values for the previous time window used for consistency constraints
    @ Out, win_opt_dispatch, BlackboxDispatch, the optimal dispatch over the time_window
    @ Out, store_lvl, storage levels of the storage components
    @ Out, sol.fun 
    """
    print(f'solving window: {start_i}-{end_i}')
    window_length = len(time_window)

    # Step 1) Build the resource pool constraint functions
    pool_cons = self._build_pool_cons(time_window, start_i, end_i, init_store)

    # Step 2) Set up the objective function and constraint functions
    objective = self.generate_objective()

    obj_scale = 1.0
    if self.scale_objective:
      # Make an initial call to the objective function and scale it
      init_dispatch = {}
      for comp in self.components:
        if comp.dispatch_type != 'fixed':
          init_dispatch[comp.name] = comp.guess[start_i:end_i]

      # get the initial dispatch so it can be used for scaling
      initdp, _ = self.determine_dispatch(init_dispatch, time_window, start_i, end_i, init_store)
      obj_scale = objective(initdp)

    # Figure out the initial storage levels
    # if this is the first time window, use the 'storage_init_level' property.
    # Otherwise use the end storage level from the previous time window.
    storage_levels = {}
    for comp in self.components:
      if comp.stores:
        if start_i == 0:
          storage_levels[comp.name] = comp.storage_init_level
        else:
          storage_levels[comp.name] = self.storage_levels[comp.name][start_i-1]

# Make a map of the available storage elements for each resource
    storage_dict = {res: [] for res in self.resources}
    for c in self.components:
      if c.stores:
        storage_dict[c.stores].append(c)

    # Get the full dispatch based on the initial guess
    guess_dispatch = {}
    for i, c in enumerate(self.components):
      if c.dispatch_type != 'fixed':
        guess_dispatch[c.name] = c.guess
    guess_dispatch, store_lvl = self.determine_dispatch(
                                    guess_dispatch, time_window, start_i, end_i, init_store)

    # Determine what resource balances errors this dispatch would result in
    resource_errors = {}
    for res in self.resources:
      err = np.zeros(window_length)
      for c in self.components:
        if res in c.get_resources():
          if c.dispatch_type != 'fixed':
            err += guess_dispatch.state[c.name][res][start_i:end_i]
          else:
            err += c.guess[start_i: end_i]
      resource_errors[res] = err

    # Split the resource balance errors between the storage components for each resource
    # The goal is not to find the optimum, but to be as likely as possible to start with
    # a guess in the feasible region.
    for res in storage_dict:
      for s in storage_dict[res]:
        for t in range(window_length):
          dguess = max(min(resource_errors[res][t], s.ramp_rate_up[t+start_i]), -s.ramp_rate_down[t+start_i])
          s.guess[t+start_i] = dguess
          resource_errors[res][t] -= dguess

    def obj(x: List[float]) -> float:
      """
      @ In, x, a list of floats
      @ Out, scaled objective of the dispatch
      """
      # Unpack the array into a dict
      x_dict = {}
      for i, c in enumerate(self.components):
        if c.dispatch_type != 'fixed':
          x_dict[c.name] = np.array(x[i*window_length:(i+1)*window_length])
      dispatch, _ = self.determine_dispatch(x_dict, time_window, start_i, end_i, init_store)
      return objective(dispatch)/obj_scale

    # Step 3) Formulate the problem for IPOPT
    guess = []
    constraints = []
    bounds = []
    # Indexes into the array of optimization variables, skips the fixed dispatch components
    array_indexer = 0
    for comp in self.components:
      # Fixed dispatch components do not contribute to the variables or constraints of the optimization
      if comp.dispatch_type == 'fixed':
        continue

      ramping_constraints = self._gen_ramp_constraints(comp, array_indexer, window_length, start_i, prev_win_end)
      constraints.extend(ramping_constraints)

      if comp.stores:
        # min_capacity = comp.min_capacity[start_i:end_i]
        # max_capacity = comp.capacity[start_i:end_i]
        ramp_up = comp.ramp_rate_up[start_i:end_i]
        ramp_down = -1*comp.ramp_rate_down[start_i:end_i]

          # def storage_capacity_lower():
          #     dispatch, store_lvl = self.determine_dispatch(inpt, time_window, start_i, end_i, init_store)
          #     # FIXME: Check that the storage is within bounds
          # constraints.append({
          #     'type': 'ineq',
          #     'fun': storage_capacity_lower})
          # def storage_capacity_upper():
          #     dispatch, store_lvl = self.determine_dispatch(inpt, time_window, start_i, end_i, init_store)
          #     # FIXME: Check that the storage is within bounds
          # constraints.append({
          #     'type': 'ineq',
          #     'fun': storage_capacity_upper})
        bounds.extend([(ramp_down[idx], ramp_up[idx]) for idx in range(window_length)])
        guess.extend(np.zeros(window_length))
      else:
        bounds.extend((comp.min_capacity[i+start_i], comp.capacity[i+start_i]) for i in range(window_length))
        guess.extend(comp.guess[start_i:end_i])
      array_indexer += 1

    for cons in pool_cons:
      constraints.append({
          'fun': cons,
          'type': 'eq'
      })

    # # Step 4) Run the optimization
    ipopt_options = {
      'option_file_name': '',
      'maxiter': 10000,
      'tol': 1e-2, # This needs to be fairly loose to allow problems to solve
      'expect_infeasible_problem': 'yes',
      'jacobian_approximation': 'finite-difference-values',
      # 'gradient_approximation': 'finite-difference-values'
    }
    print('Initial Constraint Status: ', [c['fun'](guess) for c in constraints])
    sol = minimize_ipopt(obj, guess,
                    bounds=bounds, constraints=constraints, options=ipopt_options, tol=1e-5)
    print('Final Constraint Status: ', [c['fun'](sol.x) for c in constraints])
    print(sol.message)

    # FIXME: Try making the penalty for the slack storage less intense
    # FIXME: Focus on variable intialization. If we can get the two test cases working reliably enough
    # that may be enough for the project

      # Step 5) Set the activities on each component
    opt_dispatch = {}
    # Need to skip over idexing the non-dispatched components
    i = 0
    for c in self.components:
      if c.dispatch_type != 'fixed':
        opt_dispatch[c.name] = sol.x[window_length*i:window_length*(i+1)]
        i += 1

    win_opt_dispatch, store_lvl = self.determine_dispatch(
                                    opt_dispatch, time_window, start_i, end_i, init_store)
    return win_opt_dispatch, store_lvl, sol.fun

  def gen_slack_storage_trans(self, _) -> callable:
    """
    A closure that returns trans function
    @ In, none
    @ Out, trans
    """
    def trans(inputs, init_store):
      """
      @ In, inputs
      @ In, init_store, initial storage level
      @ Out,
      """
      tmp = np.insert(inputs, 0, init_store)
      return np.cumsum(tmp)[1:]
    return trans

  def gen_slack_storage_cost(self, res) -> callable:
    """
    A closure that returns the cost function
    @ In, res
    @ Out, cost
    """
    def cost(dispatch):
      """
      @ In, dispatch
      @ Out, summation of the cost of the dispatch
      """
      return np.sum(1e6*dispatch[res])
    return cost

  def add_slack_storage(self) -> None:
    """
    @ In, none
    @ Out, none
    """
    for res in self.resources:
      num = 1e6*np.ones(len(self.time))
      guess = np.zeros(len(self.time))

      transfer_fn = self.gen_slack_storage_trans(res)
      cost_fn = self.gen_slack_storage_cost(res)

      c = BlackboxComponent(f'{res}_slack', num, num, -num, res, transfer_fn,
                                    cost_fn, stores=res, guess=guess)
      self.components.append(c)
    self.slack_storage_added = True

  def dispatch(self, components: List[BlackboxComponent],
                time: List[float], external_obj_func: callable=None, meta=None,
                verbose: bool=False, scale_objective: bool=True,
                slack_storage: bool=False) -> BlackboxSolution:
    """
    Optimally dispatch a given set of components over a time horizon
    using a list of TimeSeries
    @ In, components, List of components to dispatch
    @ In, time, list, time horizon to dispatch the components over
    @ In, external_obj_func, callable, An external objective function
    @ In, meta, stuff, an arbitrary object passed to the transfer functions
    @ In, verbose, bool, Whether to print verbose dispatch
    @ In, scale_objective, bool, Whether to scale the objective function by its initial value
    @ In, slack_storage, bool, Whether to use artificial storage components as "slack" variables
    @ Out, optDispatch, A dispatch-state object representing the optimal system dispatch
    @ Out storage_levels, the storage levels of the system components
    Note that use of `external_obj_func` will replace the use of all component cost functions
    """
    self.components = components
    self.time = time
    self.verbose = verbose
    self.scale_objective = scale_objective
    self.external_obj_func = external_obj_func # Should be callable or None
    self.meta = meta

    resources = [c.get_resources() for c in self.components]
    self.resources = list(set(chain.from_iterable(resources)))

    if slack_storage and not self.slack_storage_added:
      self.add_slack_storage()

    return self._dispatch_pool()

try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils

def convert_dispatch(ch_dispatch: BlackboxDispatchState, resource_map: dict,
                      component_map: dict) -> NumpyState:
  """
  Convert a Chickadee dispatch object to a NumpyState object
  @ In, ch_dispatch, chickadee Dispatch, The dispatch to convert
  @ In, resource_map, dict, HERON resource map
  @ In, component_map, dict, a map of Chickadee components to HERON ones
  @ Out, np_dispatch, NumpyState, The converted dispatch
  """

  # This just needs to be a reliable unique identifier for each component.
  # In HERON it is a HERON Component. Here we just use the component names.
  np_dispatch = NumpyState()

  np_dispatch.initialize(component_map.values(), resource_map, ch_dispatch.time)

  start_i = 0
  end_i = len(ch_dispatch.time)

  # Copy over all the activities
  for c, data in ch_dispatch.state.items():
    for res, values in data.items():
      np_dispatch.set_activity_vector(component_map[c], res,
              start_i, end_i, values)

  return np_dispatch

def generate_transfer(comp, sources, dt):
  """
  @ In, comp
  @ In, sources
  @ In, dt
  @ Out, transfer
  """
  interaction = comp.get_interaction()
  if comp._stores:
    # For storage components there is no transfer, so the transfer method is attached to the
    # rate node instead.
    transfer = interaction._rate._obj._module_methods[interaction._rate._sub_name]
    if transfer is None:
      raise Exception(f'A Storage component ({comp.name}) cannot be defined without a transfer function when using the Chickadee dispatcher.')
    return transfer
  else:
    thing = interaction._transfer
    if thing is None:
      # For components that don't actually transfer, HERON never loads the functions
      return lambda x: {}
  # We really need to dig for this one, but it lets us determine our own
  # function signatures for the transfer functions by bypassing the HERON interfaces
  transfer = interaction._transfer._obj._module_methods[thing._sub_name]
  return transfer

class BlackBoxDispatcher(Dispatcher):
  """
  Dispatch using pyOptSparse optimization package through Chickadee
  """

  @classmethod
  def get_input_specs(cls):
    """
    @ In, cls
    @ Out, specs
    """
    specs = InputData.parameterInputFactory('blackbox', ordered=False, baseNode=None)
    return specs

  def __init__(self):
    """
    @ In, none
    @ Out, none
    """
    try:
      from cyipopt import minimize_ipopt
    except:
      raise ModuleNotFoundError('Optional cyipopt dependency required. Please install the optional dependencies')

    self.name = 'BlackboxDispatcher'

  def dispatch(self, case, components, sources, meta):
    """
      Dispatch the system using IPOPT, pyOptSparse and Chickadee.
      @ In, case, Case
      @ In, components, List[Component], the system components
      @ In, sources
      @ In, meta
      @ Out, opt_dispatch, NumpyState, the Optimal system dispatch
    """
    # Get the right time horizon
    time_horizon = np.linspace(*self.get_time_discr())
    # This is needed for HERON to pull handle the ARMAs properly later on
    meta['HERON']['time_index'] = slice(0, len(time_horizon))
    dt = time_horizon[1] - time_horizon[0]

    resource_map = meta['HERON']['resource_indexer']

    # Convert the components to Chickadee components
    ch_comps = []
    comp_map = {}
    for c in components:
      tf = c.get_interaction()._transfer
      capacity_var = c.get_capacity_var()
      print(c.name, capacity_var)
      cap = c.get_capacity(meta)[0][capacity_var]
      capacity = np.ones(len(time_horizon)) * cap
      ch_comp = BlackboxComponent(
        c.name,
        capacity,
        1e5*np.ones(len(time_horizon)),
        1e5*np.ones(len(time_horizon)),
        capacity_var,
        generate_transfer(c, sources, dt),
        None, # External cost function is used
        produces=list(c.get_outputs()),
        consumes=list(c.get_inputs()),
        # It turns out c._stores only holds a <HERON Storage> object,
        # so we get the resource from the inputs
        stores=list(c.get_inputs())[0] if c._stores else None,
        dispatch_type=c.is_dispatchable()
      )
      ch_comps.append(ch_comp)
      comp_map[ch_comp.name] = c

    # Make the objective function
    def objective(dispatchState: BlackboxDispatchState):
    #print(len(dispatchState.time), {key: { res: len(d) for res, d in dispatchState.state[key].items()}for key in dispatchState.state.keys()})
      np_dispatch = convert_dispatch(dispatchState, resource_map, comp_map)
      return self._compute_cashflows(components, np_dispatch,
                                      dispatchState.time, meta)

    # Dispatch using Chickadee
    dispatcher = ChickadeeDispatcher()
    solution = dispatcher.dispatch(ch_comps, time_horizon, meta=meta,
                                        external_obj_func=objective)

    # Convert Chickadee dispatch back to HERON dispatch for return
    solution_dispatch = BlackboxDispatchState(ch_comps, time_horizon)
    solution_dispatch.state = solution.dispatch
    return convert_dispatch(solution_dispatch, resource_map, comp_map)

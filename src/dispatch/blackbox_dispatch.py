from .Dispatcher import Dispatcher
#from .Component import PyOptSparseComponent
from .Component import PyomoBlackboxComponent
from .TimeSeries import TimeSeries
from .Solution import Solution

# Don't import pyoptsparse here as Gekko users may not want to install it
# import pyoptsparse
from pyomo.environ import *
import pyomo
from pyomo.util.model_size import build_model_size_report

import numpy as np
import sys
import os
import time as time_lib
import traceback
from typing import List
from itertools import chain
from pprint import pformat
from copy import deepcopy

class DispatchState(object):
    '''Modeled after idaholab/HERON NumpyState object'''
    def __init__(self, components: List[PyomoBlackboxComponent], time: List[float]):
        s = {}

        for c in components:
            s[c.name] = {}
            for resource in c.get_resources():
                s[c.name][resource] = np.zeros(len(time))

        self.state = s
        self.time = time

    def set_activity(self, component: PyomoBlackboxComponent, resource, activity, i=None):
        if i is None:
            self.state[component.name][resource] = activity
        else:
            self.state[component.name][resource][i] = activity

    def get_activity(self, component: PyomoBlackboxComponent, resource, i=None):
        try:
            if i is None:
                return self.state[component.name][resource]
            else:
                return self.state[component.name][resource][i]
        except Exception as err:
            print(i)
            raise err

    def set_activity_vector(self, component: PyomoBlackboxComponent,
                            resource, start, end, activity):
        self.state[component.name][resource][start:end] = activity

    def __repr__(self):
        return pformat(self.state)


class PyomoBlackbox(Dispatcher):
    '''
    Dispatch using Pyomo optimization package and a pool-based method.
    '''

    slack_storage_added = False # In case slack storage is added in a loop

    def __init__(self, window_length=10):
        try:
            # Import this here, so it is not loaded unless the pyOptSparse
            # dispatcher is actually being used. Allows using Chickadee with
            # Gekko without the need to install pyOptSparse
            import pyoptsparse
        except:
            print('Error importing pyOptSparse. This module must be installed when the pyOpySparse dispatcher is used.')

        self.name = 'PyOptSparseDispatcher'
        self._window_length = window_length

        # Defined on call to self.dispatch
        self.components = None
        self.case = None

    def _gen_pool_cons(self, resource) -> callable:
        '''A closure for generating a pool constraint for a resource
        :param resource: the resource to evaluate
        :returns: a function representing the pool constraint
        '''

        def pool_cons(dispatch_window: DispatchState) -> float:
            '''A resource pool constraint
            Checks that the net amount of a resource being consumed, produced and
            stored is zero.
            :param dispatch_window: the dispatch to evaluate
            :returns: SSE of resource constraint violations
            '''
            time = dispatch_window.time
            n = len(time)
            err = np.zeros(n)

            # FIXME: This is an inefficient way of doing this. Find a better way
            cs = [c for c in self.components if resource in c.get_resources()]
            for i, _ in enumerate(time):
                for c in cs:
                    if c.stores:
                        err[i] += -dispatch_window.get_activity(c, resource, i)
                    else:
                        err[i] += dispatch_window.get_activity(c, resource, i)

            # FIXME: This simply returns the sum of the errors over time. There
            # are likely much better ways of handling this.
            # Maybe have a constraint for the resource at each point in time?
            # Maybe use the storage components as slack variables?
            return sum(err**2)

        return pool_cons

    def _build_pool_cons(self) -> List[callable]:
        '''Build the pool constraints
        :returns: List[callable] a list of pool constraints, one for each resource
        '''

        cons = []
        for res in self.resources:
            # Generate the pool constraint here
            pool_cons = self._gen_pool_cons(res)
            cons.append(pool_cons)
        return cons

    def determine_dispatch(self, opt_vars: dict, time: List[float],
                            start_i: int, end_i: int, init_store: dict) -> DispatchState:
        '''Determine the dispatch from a given set of optimization
        vars by running the transfer functions. Returns a Numpy dispatch
        object
        :param opt_vars: dict, holder for all the optimization variables
        :param time: list, time horizon to dispatch over
        :param start_i:
        :param end_i:
        :param init_store:
        :returns: DispatchState, dispatch of the system
        :returns: dict, storage levels of each storage component over time
        '''
        # Initialize the dispatch
        dispatch = DispatchState(self.components, time)
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

    def _dispatch_pool(self) -> Solution:
        '''Dispatch the given system using a resource-pool method
        :returns: DispatchState, the optimal dispatch of the system
        :returns: dict, the storage levels of the storage components

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
        '''

        self.start_time = time_lib.time()
        objval = 0.0

        # Step 1) Find the vars: 1 for each component input where dispatch is not fixed
        self.vs = {}  # Min/Max tuples of the various input
        self.storage_levels = {}
        for c in self.components:
            if c.dispatch_type == 'fixed':
                # Fixed dispatch components do not contribute to the variables
                continue
            else:  # Independent and dependent dispatch
                lower = c.min_capacity
                upper = c.capacity
                # Note: This assumes everything based off the first point
                if lower[0] < upper[0]:
                    self.vs[c.name] = [lower, upper]
                else:
                    self.vs[c.name] = [upper, lower]
            if c.stores:
                self.storage_levels[c.name] = np.zeros(len(self.time))

        full_dispatch = DispatchState(self.components, self.time)

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
        solution = Solution(self.time, full_dispatch.state, self.storage_levels,
                                False, objval, time_windows=time_windows)
        return solution

    def generate_objective(self) -> callable:
        '''Assembles an objective function to minimize the system cost'''
        if self.external_obj_func:
            return self.external_obj_func
        else:

            def objective(dispatch: DispatchState) -> float:
                '''The objective function. It is broken out to allow for easier scaling.
                :param dispatch: the full dispatch of the system
                :returns: float, value of the objective function
                '''
                obj = 0.0
                for c in self.components:
                    obj += c.cost_function(dispatch.state[c.name])
                print('Returning: ', obj)
                return pyomo.core.expr.numvalue.NumericConstant(obj)
            return objective

    def _dispatch_window(self, time_window: List[float], start_i: int,
                        end_i: int, init_store, prev_win_end: dict=None) -> Solution:
        '''Dispatch a time-window using a resource-pool method
        :param time_window: The time window to dispatch the system over
        if !u
        :param start_i: The time-array index for the start of the window
        :param end_i: The time-array index for the end of the window
        :param init_store: dict of the initial storage values of the storage components
        :param prev_win_end: dict of the ending values for the previous time window used for consistency constraints
        :returns: DispatchState, the optimal dispatch over the time_window
        '''

        # Step 1) Build the resource pool constraint functions
        pool_cons = self._build_pool_cons()

        # Step 2) Set up the objective function and constraint functions
        objective = self.generate_objective()

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

        for c in self.components:
            if c.stores:
                resource = c.stores
                # Find the resource error between other generators, storers, and consumers of the resource
                # Assign the guess value of the storage component to cover the difference. 
                c.guess = ...

        # Step 3) Formulate the problem for pyomo
        #optProb = pyoptsparse.Optimization('Dispatch', optimize_me)
        model = ConcreteModel()

        window_length = len(time_window)

        model.T = Set(initialize=np.arange(0, window_length, dtype =int))

        for comp in self.components:
            if comp.dispatch_type != 'fixed':
                bounds = [bnd[start_i:end_i] for bnd in self.vs[comp.name]]
                guess = {}
                for i in range(len(time_window)):
                    guess[i]= comp.guess[start_i+i]
                ramp_up = comp.ramp_rate_up[start_i:end_i]
                ramp_down = comp.ramp_rate_down[start_i:end_i]

                if comp.stores: #This is not tested when debugging our test (yet)
                    min_capacity = comp.min_capacity[start_i:end_i]
                    max_capacity = comp.capacity[start_i:end_i]
                    ramp_up = comp.ramp_rate_up[start_i:end_i]
                    ramp_down = -1*comp.ramp_rate_down[start_i:end_i]
                    # Storage Variable
                    # Generate the lower and upper storage rates for storage components
                    bounds_generator = lambda m, j: (ramp_down[j], ramp_up[j])
                    initvals_generator = lambda m, j: guess[j]
                    # Generate and set the Pyomo variable for the component output
                    comp_var = Var(range(window_length), initialize=initvals_generator, bounds=bounds_generator)
                    setattr(model, comp.name, comp_var)

                else:
                    # Generate the lower and upper capacity bounds for non-storage components
                    bounds_generator = lambda m, j: (bounds[0][j], bounds[1][j])
                    # Generate the initial values for non-storage components
                    initvals_generator = lambda m, j: guess[j]
                    # Generate and set the Pyomo variable for the component output
                    comp_var = Var(range(window_length), initialize=initvals_generator, bounds=bounds_generator)
                    setattr(model, comp.name, comp_var)

                # Ramp Constraints for both storage and non-storage components        
                if start_i == 0:
                    for t in range(window_length-1):
                        comp_ramp_down_constr = Constraint(expr=(comp_var[t+1]-comp_var[t]) <= comp.ramp_rate_up[t])
                        setattr(model, f'{comp.name}_ramp_down_{t}', comp_ramp_down_constr)


                        comp_ramp_up_constr = Constraint(expr=(comp_var[t+1]-comp_var[t]) >= -1*comp.ramp_rate_down[t])
                        setattr(model, f'{comp.name}_ramp_up_{t}', comp_ramp_up_constr)

                else:
                    comp_ramp_down_constr = Constraint(expr=(comp_var[0]-prev_win_end[comp.name]) <= comp.ramp_rate_up[start_i])
                    setattr(model, f'{comp.name}_ramp_down_0', comp_ramp_down_constr)


                    comp_ramp_up_constr = Constraint(expr=(comp_var[0]-prev_win_end[comp.name]) >= -1*comp.ramp_rate_down[start_i])
                    setattr(model, f'{comp.name}_ramp_up_0', comp_ramp_up_constr)
                    for t in range(window_length-1):
                        comp_ramp_down_constr = Constraint(expr=(comp_var[t+1]-comp_var[t]) <= comp.ramp_rate_up[start_i+t])
                        setattr(model, f'{comp.name}_ramp_down_{t+1}', comp_ramp_down_constr)


                        comp_ramp_up_constr = Constraint(expr=(comp_var[t+1]-comp_var[t]) >= -1*comp.ramp_rate_down[start_i+t])
                        setattr(model, f'{comp.name}_ramp_up_{t+1}', comp_ramp_up_constr)

        # Storage Level Constraint (Not sure how to go about this)
        def storage_wrapper(*args):
            '''Wraps the objective function so that it can take a single array as input'''
            stuff_dict = {}
            # Unpack the array into a dict
            for i, c in enumerate(self.components):
                if c.dispatch_type != 'fixed':
                    stuff_dict[c.name] = np.array(args[i*window_length:(i+1)*window_length])
            dispatch, store_lvl = self.determine_dispatch(stuff_dict, time_window, start_i, end_i, init_store)
            storage_violation = 0.0
            for i, c in enumerate(self.components):
                if c.stores:
                    upr_violations = np.where(store_lvl[c.name] > c.capacity[0])
                    lwr_violations = np.where(store_lvl[c.name] < c.min_capacity[0])
                    storage_violation += np.sum(upr_violations)
                    storage_violation += np.sum(lwr_violations)

            return storage_violation
        def storage_gradient(stuff, fixed=False, step=1e-6):
            c = storage_wrapper(stuff)
            grad = []
            for i, v in enumerate(stuff):
                stuff_copy = deepcopy(stuff)
                stuff_copy[i] = v+step
                grad.append(storage_wrapper((stuff_copy - c)/step))
            return grad
        
        model.ext_storage_fn = ExternalFunction(storage_wrapper, storage_gradient)
        inpt=[]
        for c in self.components:
            if c.dispatch_type == 'fixed':
                continue
            inpt.extend(getattr(model, c.name))

        storage_constr = Constraint(expr=model.ext_storage_fn(*inpt)<=5)
        setattr(model, f'storage_constr', storage_constr)

        def objective_wrapper(*args):
            '''Wraps the objective function so that it can take a single array as input'''
            stuff_dict = {}
            # Unpack the array into a dict
            for i, c in enumerate(self.components):
                if c.dispatch_type != 'fixed':
                    stuff_dict[c.name] = np.array(args[i*window_length:(i+1)*window_length])

            dispatch, store_lvl = self.determine_dispatch(stuff_dict, time_window, start_i, end_i, init_store) 
            return float(objective(dispatch))

        def objective_gradient(stuff, fixed=False, step=1e-6):
            c = objective_wrapper(stuff)
            grad = []
            for i, v in enumerate(stuff):
                stuff_copy = deepcopy(stuff)
                stuff_copy[i] = v+step
                grad.append(objective_wrapper((stuff_copy - c)/step))
            return grad
        
        def resource_constr_wrapper(*args):
            '''Wraps the objective function so that it can take a single array as input'''
            stuff_dict = {}
            # Unpack the array into a dict
            for i, c in enumerate(self.components):
                if c.dispatch_type != 'fixed':
                    stuff_dict[c.name] = np.array(args[i*window_length:(i+1)*window_length])

            dispatch, store_lvl = self.determine_dispatch(stuff_dict, time_window, start_i, end_i, init_store) 
            return sum([cons(dispatch) for cons in pool_cons])
        
        def resource_constr_gradient(stuff, fixed=False, step=1e-6):
            c = resource_constr_wrapper(stuff)
            grad = []
            for i, v in enumerate(stuff):
                stuff_copy = deepcopy(stuff)
                stuff_copy[i] = v+step
                grad.append(resource_constr_wrapper((stuff_copy - c)/step))
            return grad

        model.ext_resource_constr_fn = ExternalFunction(resource_constr_wrapper,resource_constr_gradient)
        model.ext_objective_fn = ExternalFunction(objective_wrapper, objective_gradient)
        # pack the various variables into a single array in a predictable manner
        inpt = []
        # This creates a single array of all the problem variables
        # as that is the form needed by Pyomo
        for c in self.components:
            if c.dispatch_type == 'fixed':
                continue
            inpt.extend(getattr(model, c.name))
            # inpt.extend(getattr(model, f'{c.name}_ramp'))
        model.ResourceConstr = Constraint(expr=model.ext_resource_constr_fn(*inpt)<=100)
        model.MyObj = Objective(expr=model.ext_objective_fn(*inpt), sense=minimize)

        # Step 4) Run the optimization
        print('------   About to solve  ------')
        optimizer = SolverFactory('trustregion', verbose=True)
        #optimizer.options['solver'] = {'expect_infeasible_problem': 'yes'}
        # raise Exception('Stopping here for right now...')
        dofs = []
        # optimizer.config.solver = {'expect_infeasible_problem': 'yes'}
        
        for c in self.components:
            if c.dispatch_type == 'fixed':
                continue
            # NOTE: CANNOT splat/spread the sets as it destroys their type
            for t in range(window_length):
                dofs.append(getattr(model, c.name)[t])

        results = optimizer.solve(model, dofs, tee=True) #, solver = {'expect_infeasible_problem': 'yes'})
        inpt = []
        for c in self.components:
            if c.dispatch_type == 'fixed':
                continue
            comp_var = getattr(model, c.name)
            inpt.extend(comp_var)
        window_cost = value(model.MyObj)

        print('------   Finished solve  ------')
        try:

            model.display()
            # How do I extract all of the needed information from the solved pyomo model
            # FIXME: Find a way of returning failed windows
            ...
        except Exception as err:
            print('Dispatch optimization failed:')
            traceback.print_exc()
            raise err

        opt_dispatch = {}
        for i, c in enumerate(self.components):
            if c.dispatch_type != 'fixed':
                opt_dispatch[c.name] = np.array(list(getattr(results, c.name).get_values().values()))

        # Step 5) Set the activities on each component
        win_opt_dispatch, store_lvl = self.determine_dispatch(
                                        opt_dispatch, time_window, start_i, end_i, init_store)
        return win_opt_dispatch, store_lvl, window_cost

    def gen_slack_storage_trans(self, res) -> callable:
        def trans(data, meta):
            return data, meta
        return trans

    def gen_slack_storage_cost(self, res) -> callable:
        def cost(dispatch):
            return np.sum(1e10*dispatch[res])
        return cost

    def add_slack_storage(self) -> None:
        for res in self.resources:
            num = 1e10*np.ones(len(self.time))
            guess = np.zeros(len(self.time))

            trans = self.gen_slack_storage_trans(res)
            cost = self.gen_slack_storage_cost(res)

            c = PyomoBlackboxComponent(f'{res}_slack', num, num, num, res, trans,
                                        cost, stores=[res], guess=guess)
            self.components.append(c)
            self.slack_storage_added = True

    def dispatch(self, components: List[PyomoBlackboxComponent],
                    time: List[float], timeSeries: List[TimeSeries] = [],
                    external_obj_func: callable=None, meta=None,
                    verbose: bool=False, scale_objective: bool=True,
                    slack_storage: bool=False) -> Solution:
        """Optimally dispatch a given set of components over a time horizon
        using a list of TimeSeries

        :param components: List of components to dispatch
        :param time: time horizon to dispatch the components over
        :param timeSeries: list of TimeSeries objects needed for the dispatch
        :param external_obj_func: callable, An external objective function
        :param meta: stuff, an arbitrary object passed to the transfer functions
        :param verbose: Whether to print verbose dispatch
        :param scale_objective: Whether to scale the objective function by its initial value
        :param slack_storage: Whether to use artificial storage components as "slack" variables
        :returns: optDispatch, A dispatch-state object representing the optimal system dispatch
        :returns: storage_levels, the storage levels of the system components
        Note that use of `external_obj_func` will replace the use of all component cost functions
        """
        # FIXME: Should check to make sure that the components have arrays of the right length
        self.components = components
        self.time = time
        self.verbose = verbose
        self.timeSeries = timeSeries
        self.scale_objective = scale_objective
        self.external_obj_func = external_obj_func # Should be callable or None
        self.meta = meta

        resources = [c.get_resources() for c in self.components]
        self.resources = list(set(chain.from_iterable(resources)))

        if slack_storage and not self.slack_storage_added:
            self.add_slack_storage()

        return self._dispatch_pool()

# ToDo:
# - Try priming the initial values for generic systems better
# - Calculate exact derivatives using JAX if possible. Pyomo may already do this for us.
# - Handle infeasible cases clearly. Raise an error if the constraints are not met.
# - Need to add a method to catch if user transfer functions provide the right responses

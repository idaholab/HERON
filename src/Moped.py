#!/usr/bin/env python
# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Alternative analysis approach to HERON's standard RAVEN running RAVEN, contains all the necessary methods to run
  a monolithic solve that utilizes TEAL cashflows, RAVEN ROM(s), and pyomo optimization.
"""
import os
import sys

from functools import partial
import itertools as it
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np

import _utils as hutils
# Getting raven location
path_to_raven = hutils.get_raven_loc()
# Access to externalROMloader
sys.path.append(os.path.abspath(os.path.join(path_to_raven, 'scripts')))
# Access to TEAL
sys.path.append(os.path.abspath(os.path.join(path_to_raven, 'plugins')))
# General access to RAVEN
sys.path.append(path_to_raven)
from HERON.src.base import Base
from ravenframework.MessageHandler import MessageHandler
import externalROMloader as ROMloader
from TEAL.src import CashFlows
from TEAL.src import main as RunCashFlow

class MOPED(Base):
    def __init__(self):
        super().__init__()
        self._components = []                 # List of components objects from heron input
        self._sources = []                    # List of sources objects from heron input
        self._case = None                     # Case object that contains the case parameters
        self._econ_settings = None            # TEAL global settings used for building cashflows
        self._m = None                        # Pyomo model to be solved
        self._producers = []                  # List of pyomo var/params of producing components
        self._eval_mode = 'clustered'         # (full or clustered) clustered is better for testing and speed, full gives a more realistic NPV result
        self._yearly_hours = 24*365           # Number of hours in a year to handle dispatch, based on clustering
        self._component_meta = {}             # Primary data structure for MOPED, organizes important information for problem construction
        self._cf_meta = {}                    # Secondary data structure for MOPED, contains cashflow info
        self._resources = []                  # List of resources used in this analysis
        self._solver = SolverFactory('ipopt') # Solver for optimization solve, default is 'ipopt', TODO allow user to specify
        self._cf_components = []              # List of TEAL.Components objects generated for analysis
        self._dispatch = []                   # List of pyomo vars/params for each realization and year

        self.messageHandler = MessageHandler()

    def buildActivity(self):
        """
          Generate active list that is necessary for building TEAL settings object
          @ In, None
          @ Out, activity, list, associates component with cashflow types ([ngcc|'Cap', ngcc|'Hourly'])
        """
        activity = []
        for comp in self._components:
            #TODO Does this need to be expanded on?
            for cf in comp._economics._cash_flows:
                if cf._type == 'one-time':
                    activity.append(f'{comp.name}|Capex')
                elif cf._type == 'repeating':
                    if cf._period =='year':
                        activity.append(f'{comp.name}|Yearly')
                        continue
                    # Necessary to have activity indicator account for multiple dispatch realizations
                    for real in range(self._case._num_samples):
                        activity.append(f'{comp.name}|Dispatching_{real+1}')
        self.raiseADebug(f'Built activity Indicator: {activity}')
        return activity

    def buildEconSettings(self, verbosity=0):
        """
          Builds TEAL economic settings for running cashflows
          @ In, verbosity, int or string, verbosity settings for TEAL
          @ out, None
        """
        activity = self.buildActivity()
        params = self._case._global_econ
        params['Indicator']['active'] = activity
        params['verbosity'] = verbosity
        self.raiseADebug('Building economic settings...')
        valid_params = ['ProjectTime', 'DiscountRate',
                        'tax', 'inflation', 'verbosity', 'Indicator']
        for param_name, param_value in params.items():
            if param_name != 'Indicator' and param_name in valid_params:
                self.raiseADebug(f'{param_name}: {param_value}')
            elif param_name == 'Indicator':
                self.raiseADebug(f'Indicator dictionary: {params["Indicator"]}')
            else:
                raise IOError(f'{param_name} is not a valid economic setting')
        self.raiseADebug('Finished building economic settings!')
        self._econ_settings = CashFlows.GlobalSettings()
        self._econ_settings.setParams(params)

    def buildComponentMeta(self):
        """
          Build pyomo object, capacity variables, and fixed capacity parameters
          @ In, None
          @ Out, None
        """
        self._m = pyo.ConcreteModel(name=self._case.name)
        # Considering all components in analysis to build a full pyomo solve
        for comp in self._components:
            self._component_meta[comp.name] = {}
            for prod in comp._produces: # NOTE Cannot handle components that produce multiple things
                resource = prod._capacity_var
                mode = prod._capacity.type
                self.setCapacityMeta(mode, resource, comp, prod, True)
            for dem in comp._demands: # NOTE Cannot handle components that demand multiple things
                resource = dem._capacity_var
                mode = dem._capacity.type
                self.setCapacityMeta(mode, resource, comp, dem)

    def loadSyntheticHistory(self, signal):
        """
          Loads synthetic history for a specified signal,
          also sets yearly hours and pyomo indexing sets
          @ In, signal, string, name of signal to sample
          @ Out, synthetic_data, dict, contains data from evaluated ROM
        """
        # NOTE self._sources[0]._var_names are the user assigned signal names in DataGenerators
        if signal not in self._sources[0]._var_names:
            raise IOError('The requested signal name is not available'
                          'from the synthetic history, check DataGenerators node in input')
        runner = ROMloader.ravenROMexternal(self._sources[0]._target_file,
                                            hutils.get_raven_loc())
        from ravenframework.utils import xmlUtils
        inp = {'scaling':[1]}
        # TODO expand to change other pickledROM settings withing this method
        nodes = []
        node = xmlUtils.newNode('ROM', attrib={'name':'SyntheticHistory', 'subType':'pickledRom'})
        node.append(xmlUtils.newNode('clusterEvalMode', text=self._eval_mode))
        nodes.append(node)
        runner.setAdditionalParams(nodes)
        synthetic_data = {}
        for real in range(self._case._num_samples):
            self.raiseAMessage(f'Loading synthetic history for signal: {signal}')
            name = f'Realization_{real + 1}'
            current_realization = runner.evaluate(inp)[0]
            if self._eval_mode == 'full':
                # reshape so that a filler cluster index is made
                current_realization[signal] = np.expand_dims(current_realization[signal], axis = 1)
            # TODO check for multipliers other than one necessary for wind and solar at the very least
            synthetic_data[name] = current_realization[signal]
        cluster_count = synthetic_data['Realization_1'].shape[1]
        hour_count = synthetic_data['Realization_1'].shape[2]
        self._m.c = pyo.Set(initialize = np.arange(cluster_count))
        if self._eval_mode not in ['clustered', 'full']:
            raise IOError('Improper ROM evaluation mode detected, try "clustered" or "full".')
        # How many dispatch points we will have for each year
        self._yearly_hours = hour_count * cluster_count
        # TODO consider different segment lengths?
        self._m.t = pyo.Set(initialize = np.arange(hour_count))
        return synthetic_data

    def setCapacityMeta(self, mode, resource, comp, element, consumes=False):
        """
          Checks the capacity type, dispatch type, and resources involved for each component
          to build component_meta
          @ In, mode, string, type of capacity definition for component
          @ In, resource, string, resource produced or demanded
          @ In, comp, HERON component
          @ In, element, HERON produces/demands node
          @ In, consumes, bool, does this component consume resources
          @ Out, None
        """
        # Organizing important aspects of problem for later access
        if isinstance(element, type(self._components[0]._produces[0])): # Assumes first comp is a producer
        #if isinstance(type, type(self._components[0]._produces[0])):
            self._component_meta[comp.name]['Produces'] = resource
        else:
            self._component_meta[comp.name]['Demands'] = resource
        self._component_meta[comp.name]['Consumes'] = None
        self._component_meta[comp.name]['Dispatch'] = element._dispatchable
        # Different possible capacity value definitions for a component
        if mode == 'OptBounds':
            self.raiseADebug(
              f'Building pyomo capacity variable for '
              f'{comp.name}'
              )
            opt_bounds = element._capacity._vp._parametric
            # This is a capacity we make a decision on
            var = pyo.Var(initialize = 0.5*opt_bounds[1], bounds = (opt_bounds[0], opt_bounds[1]))
            setattr(self._m, f'{comp.name}', var)
        elif mode == 'SweepValues': # TODO Add capability to handle sweepvalues, maybe multiple pyo.Params?
            raise IOError('MOPED does not currently support sweep values option')
        elif mode == 'FixedValue':
            self.raiseADebug(
              f'Building pyomo capacity parameter for '
              f'{comp.name}'
              )
            # Params represent constant value components of the problem
            value = element._capacity._vp._parametric
            param = pyo.Param(initialize = value)
            setattr(self._m, f'{comp.name}', param)
        elif mode == 'SyntheticHistory':
            self.raiseADebug(
              f'Building capacity with synthetic histories for '
              f'{comp.name}'
              )
            # This method runs external ROM loader and defines some pyomo sets
            capacity = self.loadSyntheticHistory(element._capacity._vp._var_name)
            # TODO how to better handle capacities based on Synth Histories
            self._component_meta[comp.name]['Capacity'] = capacity
        if mode != 'SyntheticHistory':
            # TODO smarter way to do this check?
            self._component_meta[comp.name]['Capacity'] = getattr(self._m, f'{comp.name}')
        if consumes == True:
            # NOTE not all producers consume
            # TODO should we handle transfer functions here?
            for con in element._consumes:
                self._component_meta[comp.name]['Consumes'][con] = element._transfer

    def buildCashflowMeta(self):
        """
          Builds cashflow meta used in cashflow component construction
          @ In, None
          @ Out, None
        """
        # NOTE assumes that each component can only have one cap, yearly, and repeating
        for comp in self._components:
            self.raiseADebug(f'Retrieving cashflow information for {comp.name}')
            self._cf_meta[comp.name] = {}
            self._cf_meta[comp.name]['Lifetime'] = comp._economics._lifetime
            for cf in comp._economics._cash_flows:
                # Using reference prices for cashflows
                alpha = cf._alpha._vp._parametric
                multiplier = cf._driver._multiplier
                # Default mult should be 1
                if multiplier == None:
                    multiplier = 1
                value = multiplier * alpha
                if cf._type == 'one-time':
                    self._cf_meta[comp.name]['Capex'] = value
                    # Necessary if capex has depreciation and amortization
                    self._cf_meta[comp.name]['Deprec'] = cf._depreciate
                elif cf._type == 'repeating':
                    if cf._period == 'year':
                        self._cf_meta[comp.name]['Yearly'] = value
                        continue
                    self._cf_meta[comp.name]['Dispatching'] = value

    def createCapex(self, comp, alpha, capacity):
        """
          Builds capex TEAL cashflow for a given component
          @ In, comp, TEAL component object
          @ In, alpha, float, reference price for capex cost
          @ In, capacity, pyomo var, size of the ocmponent that drives the cost
          @ Out, cf, TEAL cashflow
        """
        life = comp.getLifetime()
        cf = CashFlows.Capex()
        cf.name = 'Capex'
        cf.initParams(life)
        cfParams = {'name': 'Capex',
                    'alpha': alpha,
                    'driver': capacity,
                    'reference': 1.0,
                    'X': 1.0,
                    'depreciate': 3,
                    'mult_target': False,
                    'inflation': None,}
        cf.setParams(cfParams)
        return cf

    def createRecurringYearly(self, comp, alpha, driver):
        """
          Constructs the parameters for capital expenditures
          @ In, comp, TEAL.src.CashFlows.Component, main structure to add component cash flows
          @ In, alpha, float, yearly price to populate
          @ In, driver, pyomo.core.base.var.ScalarVar, quantity sold to populate
          @ Out, cf, TEAL.src.CashFlows.Component, cashflow sale for the recurring yearly
        """
        # Necessary to make life integer valued for numpy
        life = int(self._case._global_econ['ProjectTime'])
        cf = CashFlows.Recurring()
        cfParams = {'name': 'Yearly',
                    'X': 1,
                    'mult_target': None,
                    'inflation': False}
        cf.setParams(cfParams)
        # 0 for first year (build year) -> TODO couldn't this be automatic?
        alphas = np.ones(life+1, dtype=object) * alpha
        drivers = np.ones(life+1, dtype=object) * driver
        alphas[0] = 0
        drivers[0] = 0
        # construct annual summary cashflows
        cf.computeYearlyCashflow(alphas, drivers)
        return cf

    def createRecurringHourly(self, comp, alpha, driver, real):
        """
          Generates recurring hourly cashflows, mostly for dispatch and sales
          @ In, comp, TEAL component
          @ In, alpha, float/np.array, reference price of sale
          @ In, driver, numpy array of pyomo.var.values that drive cost
          @ In, real, int, current realization number
          @ Out, cf, TEAL cashflow
        """
        # Necessary to make integer for numpy arrays
        life = int(self._case._global_econ['ProjectTime'])
        cf = CashFlows.Recurring()
        cfParams = {'name': f'Dispatching_{real+1}',
                    'X': 1,
                    'mult_target': False,
                    'inflation': None}
        cf.setParams(cfParams)
        cf.initParams(life, pyomoVar=True)

        # Necessary to shift year index by one since no recurring cashflows on first build year
        for year in range(life + 1):
            if isinstance(alpha, float):
                cf.computeIntrayearCashflow(year, alpha, driver[year, :])
            else:
                cf.computeIntrayearCashflow(year, alpha[year, :], driver[year, :])
        return cf

    def collectResources(self):
        """
          Searches through components to collect all resources into a list
          @ In, None
          @ Out, None
        """
        for comp in self._components:
            for prod in comp._produces:
                resource = prod._capacity_var
                if resource not in self._resources:
                    self._resources.append(resource)
                # TODO add for consuming components
            for dem in comp._demands:
                resource = dem._capacity_var
                if resource not in self._resources:
                    self._resources.append(resource)

    def buildDispatchVariables(self, comp):
        """
          Generates dispatch vars and value arrays to build components
          @ In, comp, HERON component
          @ Out, template_array, np.array, array of pyo.values used for TEAL cfs
          @ Out, capacity, np.array/pyomo.var, capacity variable for the component
        """
        # NOTE Assumes that all components will remain functional for project life
        project_life = int(self._case._global_econ['ProjectTime'])
        # Necessary to make year index one larger than project life so that year zero
        # Can be empty for recurring cashflows
        template_array = np.zeros((self._case._num_samples, project_life + 1, self._yearly_hours), dtype=object)
        capacity = self._component_meta[comp.name]['Capacity']
        dispatch_type = self._component_meta[comp.name]['Dispatch']
        # Checking for type of capacity is necessary to build dispatch variable
        self._m.dummy = pyo.Var()
        self._m.placeholder = pyo.Param()
        dummy_type = type(self._m.dummy)
        placeholder_type = type(self._m.placeholder)
        self.raiseADebug(f'Preparing dispatch container for {comp.name}...')
        for real in range(self._case._num_samples):
            for year in range(project_life):
                # TODO account for other variations of component settings, specifically if dispatchable
                if isinstance(capacity,(dummy_type, placeholder_type)):
                    # Currently independent and dependent are interchangable
                    if dispatch_type in ['independent', 'dependent']:
                        var = pyo.Var(self._m.c, self._m.t,
                        initialize=lambda m, c, t: 0,
                        domain=pyo.NonNegativeReals
                        )
                        setattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}',var)
                        # Shifting index such that year 0 remains 0
                        # Weighting each dispatch by the number of realizations (equal weight for each realization)
                        ## This corrects the NPV value
                        template_array[real, year + 1, :] = (1/self._case._num_samples)*np.array(list(var.values()))
                    elif dispatch_type == 'fixed':
                        param = pyo.Var(self._m.c, self._m.t,
                          initialize = lambda m, c, t: capacity.value,
                          domain = pyo.NonNegativeReals,)
                        setattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}',param)
                        con = pyo.Constraint(self._m.c, self._m.t,
                          expr = lambda m, c, t: param[(c,t)] == capacity)
                        setattr(self._m, f'{comp.name}_fixed_{real+1}_{year+1}',con)
                        # Shifting index such that year 0 remains 0
                        # Weighting each dispatch by the number of realizations (equal weight for each realization)
                        ## This corrects the NPV value
                        template_array[real, year + 1, :] = (1/self._case._num_samples)*np.array(list(param.values()))
                else:
                    # Currently independent and dependent are interchangable
                    if dispatch_type in ['independent', 'dependent']:
                        var = pyo.Var(self._m.c, self._m.t,
                          initialize = lambda m, c, t: 0,
                          domain = pyo.NonNegativeReals,
                          bounds = lambda m, c, t: (0, capacity[f'Realization_{real+1}'][year, c, t]))
                        setattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}',var)
                        # Shifting index such that year 0 remains 0
                        # Weighting each dispatch by the number of realizations (equal weight for each realization)
                        ## This corrects the NPV value
                        template_array[real, year + 1, :] = (1/self._case._num_samples)*np.array(list(var.values()))
                    elif dispatch_type == 'fixed':
                        param = pyo.Param(self._m.c, self._m.t,
                        initialize=lambda m, c, t: capacity[f'Realization_{real+1}'][year, c, t]
                        )
                        setattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}',param)
                        # Shifting index such that year 0 remains 0
                        # Weighting each dispatch by the number of realizations (equal weight for each realization)
                        ## This corrects the NPV value
                        template_array[real, year + 1, :] = (1/self._case._num_samples)*np.array(list(param.values()))
        return capacity, template_array

    def createCashflowComponent(self, comp, capacity, dispatch):
        """
          Builds TEAL component using pyomo dispatch and capacity variables
          @ In, capacity, pyomo.var/pyomo.param, primary driver
          @ In, life, int, number of years the component operates without replacement
          @ In, dispatch, np.array, pyomo values for dispatch variables
          @ Out, component, TEAL.Component
        """
        # Need to have TEAL component for cashflow functionality
        component = CashFlows.Component()
        params = {'name':comp.name}
        cfs = []
        cf_meta = self._cf_meta[comp.name]
        # Using read meta to evaluate possible cashflows
        for cf, value in cf_meta.items():
            if cf == 'Lifetime':
                self.raiseADebug(f'Setting component lifespan for {comp.name}')
                params['Life_time'] = value
                component.setParams(params)
            elif cf == 'Capex':
                # Capex is the most complex to handle generally due to amort
                self.raiseADebug(f'Generating Capex cashflow for {comp.name}')
                capex = self.createCapex(component, value, capacity)
                cfs.append(capex)
                depreciation = cf_meta['Deprec']
                if depreciation is not None:
                    capex.setAmortization('MACRS', depreciation)
                    amorts = component._createDepreciation(capex)
                    cfs.extend(amorts)
            # Necessary to avoid error message from expected inputs
            elif cf == 'Deprec':
                continue
            elif cf == 'Yearly':
                self.raiseADebug(f'Generating Yearly OM cashflow for {comp.name}')
                yearly = self.createRecurringYearly(component, value, capacity)
                cfs.append(yearly)
            elif cf == 'Dispatching':
                # Here value can be a np.array as well for ARMA grid pricing
                self.raiseADebug(f'Generating dispatch OM cashflow for {comp.name}')
                # Necessary to create a unique cash flow for each dispatch realization
                for real in range(self._case._num_samples):
                    var_om = self.createRecurringHourly(component, value, dispatch[real, :, :], real)
                    cfs.append(var_om)
            else:
                raise IOError(f'Unexpected cashflow type received: {cf}')
        component.addCashflows(cfs)
        return component

    def conserveResource(self, resource, real, year, M, c, t):
        """
          Generates pyomo constraints for resource conservation
          @ In, resource, string, name of resource we are conserving
          @ In, real, int, the current realization
          @ In, year, int, the current year
          @ In, M, pyomo.ConcreteModel
          @ In, c, int, index from pyomo set self._m.c
          @ In, t, int, index from pyomo set self._m.t
          @ Out, rule, boolean expression
        """
        # Initializing production and demand trackers
        produced = 0
        demanded = 0
        # Necessary to check all components involved in the analysis
        for comp in self._components:
            comp_meta = self._component_meta[comp.name]
            # Conservation constrains the dispatch decisions
            dispatch_value = getattr(self._m, f'{comp.name}_dispatch_{real + 1}_{year + 1}')
            for key, value in comp_meta.items():
                if key == 'Produces' and value == resource:
                    produced += dispatch_value[(c,t)]
                elif key == 'Demands' and value == resource:
                    demanded += dispatch_value[(c,t)]
                # TODO consider consumption and incorrect input information
        return produced == demanded

    def upper(self, comp, real, year, M, c, t):
        """
          Restricts independently dispatched compononents based on their capacity
          @ In, comp, HERON comp object
          @ In, real, int, current realization
          @ In, year, int, current year
          @ In, M, pyomo model object, MOPED pyomo ConcreteModel
          @ In, c, int, index for cluster
          @ In, t, int, index for hour within cluster
          @ Out, rule, boolean expression for upper bounding
        """
        # This is allows for the capacity to be an upper bound and decision variable
        upper_bound = getattr(self._m, f'{comp.name}')
        dispatch_value = getattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}')
        return dispatch_value[(c,t)] <= upper_bound

    def buildConstraints(self):
        """
          Builds all necessary constraints for pyomo object
          @ In, None
          @ Out, None
        """
        # Convert to int to make range() viable
        project_life = int(self._case._global_econ['ProjectTime'])
        # Type variables used for checking capacity type, based on pyomo vars
        # Defined as part of the self._m pyomo model
        dummy_type = type(self._m.dummy)
        placeholder_type = type(self._m.placeholder)
        self.raiseAMessage(f'Building necessary constraints for {self._case.name}')
        for real in range(self._case._num_samples):
            for year in range(project_life):
                # Separating constraints makes sense
                # Resource conservation
                for resource in self._resources:
                    con = pyo.Constraint(self._m.c, self._m.t,
                      rule = partial(self.conserveResource, resource, real, year))
                    setattr(self._m, f'{resource}_con_{real+1}_{year+1}', con)
                # Bounding constraints on dispatches
                for comp in self._components:
                    capacity = self._component_meta[comp.name]['Capacity']
                    if isinstance(capacity,(dummy_type, placeholder_type)):
                        con = pyo.Constraint(self._m.c, self._m.t,
                          rule = partial(self.upper, comp, real, year))
                        setattr(self._m, f'{comp.name}_upper_{real+1}_{year+1}', con)

    def solveAndDisplay(self):
        """
          Presents results of the optimization run
          @ In, None
          @ Out, None
        """
        # Results provide run times and optimizer final status
        results = self._solver.solve(self._m)
        self.raiseAMessage(f'Optimizer has finished running, here are the results\n{results}')
        for comp in self._components:
            # Not all components will have a pyomo variable
            try:
                comp_print = getattr(self._m, f'{comp.name}')
                self.raiseAMessage(f'Here is the optimized capacity for {comp.name}')
                comp_print.pprint()
            except:
                self.raiseAMessage(f'{comp.name} does not have a standard capacity')
        NPV = pyo.value(self._m.NPV)
        self.raiseAMessage(f"The final NPV is: {NPV}")

    #===========================
    # MAIN WORKFLOW
    #===========================
    def run(self):
        """
          Runs the workflow
          @ In, None
          @ Out, None
        """
        # Settings and metas help to build pyomo problem with cashflows
        self.buildEconSettings()
        self.buildComponentMeta()
        self.buildCashflowMeta()
        self.collectResources()
        # Each component will have dispatch and cashflow associated
        for comp in self._components:
            capacity, dispatch = self.buildDispatchVariables(comp)
            cf_comp = self.createCashflowComponent(comp, capacity, dispatch)
            self._cf_components.append(cf_comp)
        self.raiseAMessage(f'Building pyomo cash flow expression for {self._case.name}')
        # TEAL is our cost function generator here
        metrics = RunCashFlow.run(self._econ_settings, self._cf_components, {}, pyomoVar=True)
        self._m.NPV = pyo.Objective(expr=metrics['NPV'], sense = pyo.maximize)
        # Constraints need to be built for conservation and bounds of dispatch
        self.buildConstraints()
        # NOTE this currently displays just optimizer info and capacities and cost funtion
        # TODO does this need to present information about dispatches, how to do this?
        self.raiseAMessage(f'Running Optimizer...')
        self.solveAndDisplay()

    #===========================
    # UTILITIES
    #===========================
    def setInitialParams(self, case, components, sources):
        """
          Sets all attributes read from HERON input at once
          @ In, case, Cases.Case object
          @ In, components, list of Components.Component objects
          @ In, sources, list of Placeholders objects
          @ Out, None
        """
        self.setCase(case)
        self.setComponents(components)
        self.setSources(sources)
        self.messageHandler.initialize({'verbosity': self._case._verbosity,
                                         'callerLength': 18,
                                         'tagLength': 7,
                                         'suppressErrs': False,})
        self.raiseAMessage('Sucessfully set the input parameters for MOPED run')

    def setCase(self, case):
        """
          Sets the case attribute for the MOPED object
          @ In, case, Cases.Case object
          @ Out, None
        """
        self._case = case
        self.raiseADebug(f'Setting MOPED case variable to {case}')

    def setComponents(self, components):
        """
          Sets the components attribute for the MOPED object
          @ In, components, list of Components.Component objects
          @ Out, None
        """
        self._components = components
        self.raiseADebug(f'Setting MOPED components variable to {components}')

    def setSources(self, sources):
        """
          Sets the sources attribute for the MOPED object
          @ In, sources, list of Placeholders objects
          @ Out, None
        """
        self._sources = sources
        self.raiseADebug(f'Setting MOPED sources variable to {sources}')

    def setSolver(self, solver):
        """
          Sets optimizer that pyomo runs in MOPED
          @ In, string, solver to use
          @ Out, None
        """
        self._solver = SolverFactory(solver)
        self.raiseADebug(f'Set optimizer to be {solver}')

    def getTargetParams(self, target='all'):
        """
          Returns the case, components, and sources
          @ In, None
          @ Out, case, Cases.Case object
          @ Out, components, list of Components.Component objects
          @ Out, sources, list of Placeholder objects
        """
        case = self._case
        components = self._components
        sources = self._sources
        #TODO Expand this method to include all attributes that are useful to retrieve
        acceptable_targets = ['all', 'case', 'components', 'sources']
        if target == 'all':
            return case, components, sources
        elif target == 'case':
            return case
        elif target == 'components':
            return components
        elif target == 'sources':
            return sources
        else:
            raise IOError(f'Your {target} is not a valid attribute for MOPED.',
                          f'Please select from {acceptable_targets}')
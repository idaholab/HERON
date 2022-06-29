#!/usr/bin/env python
# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Alternative analysis approach to HERON's standard RAVEN running RAVEN, contains all the necessary methods to run
  a monolithic solve that utilizes TEAL cashflows, RAVEN ROM(s), and pyomo optimization.
"""
import os
import sys
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import _utils as hutils
import numpy as np
# Getting raven location
path_to_raven = hutils.get_raven_loc()
# Access to externalROMloader
sys.path.append(os.path.abspath(os.path.join(path_to_raven, 'scripts')))
# Access to TEAL
sys.path.append(os.path.abspath(os.path.join(path_to_raven, 'plugins')))
# General access to RAVEN
sys.path.append(path_to_raven)
import externalROMloader as ROMloader
from TEAL.src import CashFlows
import TEAL
#from TEAL.src import CashFlow as RunCashFlow

class MOPED():
    def __init__(self):
        self._components = [] # List of components objects from heron input
        self._sources = [] # List of sources objects from heron input
        self._case = None # Case object that contains the case parameters
        self._econ_settings = None # TEAL global settings used for building cashflows
        self._m = None # Pyomo model to be solved
        self._producers = [] # List of pyomo var/params of producing components
        self._eval_mode = 'clustered' # clusterEvalMode to feed the externalROMloader
        self._yearly_hours = 24*365 # Number of hours in a year to handle dispatch, based on clustering
        self._component_meta = {} # Primary data structure for MOPED,
          # organizes important information for problem construction
        self._cf_meta = {} # Secondary data structure for MOPED, contains cashflow info
        self._resources = [] # List of resources used in this analysis
        self._dispatch_variables = []
        self._roms = []
        self._verbosity = True # Verbosity setting for MOPED run
        self._solver = SolverFactory('ipopt') # Solver for optimization solve, default is 'ipopt'
        self._objective = None
        self._constraints = []

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
                    type_name = 'Cap'
                elif cf._type == 'year':
                    type_name = 'Yearly'
                elif cf._type == 'repeating':
                    type_name = 'Hourly'
                activity.append(f'{comp.name}|{type_name}')
        self.verbosityPrint(f'|Built activity Indicator: {activity}|')
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
        self.verbosityPrint('|Building economic settings...|')
        valid_params = ['ProjectTime', 'DiscountRate',
                        'tax', 'inflation', 'verbosity', 'Indicator']
        for k,v in params.items():
            if k != 'Indicator' and k in valid_params:
                self.verbosityPrint(f'|{k}: {v}|')
            elif k == 'Indicator':
                self.verbosityPrint(f'|Indicator dictionary: {params["Indicator"]}|')
            else:
                raise IOError(f'{k} is not a valid economic setting')
        self.verbosityPrint('|Finished building economic settings!|')
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
            name = f'Realization_{real + 1}'
            current_realization = runner.evaluate(inp)[0]
            # TODO check for multipliers other than one
            # Necessary for wind and solar at the very least
            synthetic_data[name] = current_realization[signal]
        # Defining pyomo indexing based off of evaluation mode
        # Necessary for including full evaluation for validation
        if self._eval_mode == 'clustered':
            cluster_count = synthetic_data['Realization_1'].shape[1]
            self._m.c = pyo.Set(initialize = np.arange(cluster_count))
        elif self._eval_mode == 'full':
            # TODO check for the number of days in the dataset instead
            cluster_count = 365
            self._m.c = pyo.Set(initialize = np.arange(cluster_count))
        else:
            raise IOError('Improper ROM evaluation mode detected, try "clustered" or "full".')
        self._yearly_hours = 24 * cluster_count
        self._m.t = pyo.Set(initialize = np.arange(24))
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
            self.verbosityPrint(
              f'|Building pyomo capacity variable for '
              f'{comp.name}|'
              )
            opt_bounds = element._capacity._vp._parametric
            # This is a capacity we make a decision on
            var = pyo.Var(initialize = 0.5*opt_bounds[1], bounds = (opt_bounds[0], opt_bounds[1]))
            setattr(self._m, f'{comp.name}', var)
        elif mode == 'SweepValues': # TODO Add capability to handle sweepvalues, maybe multiple pyo.Params?
            raise IOError('MOPED does not currently support sweep values option')
        elif mode == 'FixedValue':
            self.verbosityPrint(
              f'|Building pyomo capacity parameter for '
              f'{comp.name}|'
              )
            # Params represent constant value components of the problem
            value = element._capacity._vp._parametric
            param = pyo.Param(initialize = value)
            setattr(self._m, f'{comp.name}', param)
        elif mode == 'SyntheticHistory':
            self.verbosityPrint(
              f'|Building pyomo parameter with synthetic histories for '
              f'{comp.name}|'
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
            self.verbosityPrint(f'Retrieving cashflow information for {comp.name}')
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
                    self._cf_meta[comp.name]['Cap'] = value
                elif cf._type == 'yearly':
                    self._cf_meta[comp.name]['Yearly'] = value
                elif cf._type == 'repeating':
                    self._cf_meta[comp.name]['Hourly'] = value

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
        cf.name = 'Cap'
        cf.initParams(life)
        cfParams = {'name': 'Cap',
                    'alpha': alpha,
                    'driver': capacity,
                    'reference': 1.0,
                    'X': 1.0,
                    'depreciate': 3,
                    'mult_target': False,
                    'inflation': None,}
        cf.setParams(cfParams)
        return cf

    def createRecurringYearly(comp, alpha, driver):
        """
          Constructs the parameters for capital expenditures
          @ In, comp, TEAL.src.CashFlows.Component, main structure to add component cash flows
          @ In, alpha, float, yearly price to populate
          @ In, driver, pyomo.core.base.var.ScalarVar, quantity sold to populate
          @ Out, cf, TEAL.src.CashFlows.Component, cashflow sale for the recurring yearly
        """
        life = comp.getLifetime()
        cf = CashFlows.Recurring()
        cfFarams = {'name': 'FixedOM',
                    'X': 1,
                    'mult_target': None,
                    'inflation': False}
        cf.setParams(cfFarams)
        # 0 for first year (build year) -> TODO couldn't this be automatic?
        alphas = np.ones(life+1, dtype=object) * alpha
        drivers = np.ones(life+1, dtype=object) * driver
        alphas[0] = 0
        drivers[0] = 0
        # construct annual summary cashflows
        cf.computeYearlyCashflow(alphas, drivers)
        return cf

    def createRecurringHourly(self, comp, alpha, driver):
        """
          Generates recurring hourly cashflows, mostly for dispatch and sales
          @ In, comp, TEAL component
          @ In, alpha, float, reference price of sale
          @ In, driver, numpy array of pyomo.var.values that drive cost
          @ Out, cf, TEAL cashflow
        """
        life = comp.getLifetime()
        print('Lifetime of ', f'{comp.name} is {life}')
        cf = CashFlows.Recurring()
        cfParams = {'name': 'Hourly',
                    'X': 1,
                    'mult_target': False,
                    'inflation': None}
        cf.setParams(cfParams)
        cf.initParams(life, pyomoVar=True)
        for real in range(self._case._num_samples):
            for year in range(life):
                if isinstance(alpha, float):
                    cf.computeIntrayearCashflow(year, alpha, driver[real, year, :])
                else:
                    cf.computeIntrayearCashflow(year, alpha[real, year, :], driver[real, year, :])
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
          @ Out, None
        """
        print(self._components[0]._economics._lifetime)
        exit()
        life = comp


    def run(self):
        """
          Runs the workflow
          @ In, None
          @ Out, None
        """
        self.buildEconSettings()
        self.buildComponentMeta()
        self.buildCashflowMeta()
        self.collectResources()
        for comp in self._components:
            self.buildDispatchVariables(comp)

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
        self.verbosityPrint('|Sucessfully set the input parameters for MOPED run|')

    def setCase(self, case):
        """
          Sets the case attribute for the MOPED object
          @ In, case, Cases.Case object
          @ Out, None
        """
        self._case = case
        self.verbosityPrint(f'|Setting MOPED case variable to {case}|')

    def setComponents(self, components):
        """
          Sets the components attribute for the MOPED object
          @ In, components, list of Components.Component objects
          @ Out, None
        """
        self._components = components
        self.verbosityPrint(f'|Setting MOPED components variable to {components}|')

    def setSources(self, sources):
        """
          Sets the sources attribute for the MOPED object
          @ In, sources, list of Placeholders objects
          @ Out, None
        """
        self._sources = sources
        self.verbosityPrint(f'|Setting MOPED sources variable to {sources}|')

    def setSolver(self, solver):
        """
          Sets optimizer that pyomo runs in MOPED
          @ In, string, solver to use
          @ Out, None
        """
        self._solver = SolverFactory(solver)
        self.verbosityPrint(f'|Set optimizer to be {solver}|')

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

    def verbosityPrint(self, debugStatement):
        """
          Utility for printing information about MOPED run if desired
          @ In, debugStatement, nonspecified type, what to print
          @ Out, None
        """
        # Shorter var name
        v = self._verbosity
        if v == True or v == 1 or v == 'all':
            print('#### DEBUG MODE ####', f'{debugStatement}')
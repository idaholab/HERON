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
        self._capacity_variables = []
        self._dispatch_variables = []
        self._resources = []
        self._roms = []
        self._verbosity = True # Verbosity setting for MOPED run
        self._solver = SolverFactory('ipopt') # Solver for optimization solve, default is 'ipopt'
        self._objective = None
        self._constraints = []

    def buildActiveComponents(self):
        """
          Generate active list that is necessary for building TEAL settings object
          @ In, None
          @ Out, activity, list, associates component with cashflow types ([ngcc|'Cap', ngcc|'Hourly'])
        """
        activity = []
        for comp in self._components:
            #TODO Does this need to be expanded on?
            #TODO How to handle components without cash flows
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

    def buildEconSettings(self, verbosity = 0):
        """
          Builds TEAL economic settings for running cashflows
          @ In, verbosity, int or string, verbosity settings for TEAL
          @ out, None
        """
        activity = self.buildActiveComponents()
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

    def initializeMeta(self):
        """
          Build pyomo object, capacity variables, and fixed capacity parameters
        """
        self._m = pyo.ConcreteModel(name=self._case.name)
        component_meta = {}
        #print(self._components[0]._produces[0]._dispatchable)
        for comp in self._components:
            component_meta[comp.name] = {}
            for prod in comp._produces: #NOTE Cannot handle components that produce multiple things
                self.verbosityPrint(f'|Building pyomo capacity variable for '
                                    f'{comp.name} that produces {prod._capacity_var}|')
                resource = prod._capacity_var
                mode = prod._capacity.type
                if mode == 'OptBounds':
                    opt_bounds = prod._capacity._vp._parametric
                    var = pyo.Var(initialize = 0.5*opt_bounds[1], bounds = (opt_bounds[0], opt_bounds[1]))
                    setattr(self._m, f'{comp.name}', var)
                elif mode == 'SweepValues': #TODO Add capability to handle sweepvalues, maybe multiple pyo.Params?
                    raise IOError('MOPED does not currently support sweep values option')
                elif mode == 'FixedValue':
                    value = prod._capacity._vp._parametric
                    param = pyo.Param(initialize = value)
                    setattr(self._m, f'{comp.name}', param)
                component_meta[comp.name]['Produces'] = resource
                component_meta[comp.name]['Consumes'] = {}
                component_meta[comp.name]['Dispatch'] = prod._dispatchable
                component_meta[comp.name]['Pyomo'] = getattr(self._m, f'{comp.name}')
                for con in prod._consumes:
                    component_meta[comp.name]['Consumes'][con] = prod._transfer

            for dem in comp._demands: #NOTE Cannot handle components that produce multiple things
                self.verbosityPrint(f'|Building pyomo capacity variable for '
                                    f'{comp.name} that demands {dem._capacity_var}|')
                resource = dem._capacity_var
                mode = dem._capacity.type
                print(mode)
                exit()
                if mode == 'OptBounds':
                    opt_bounds = dem._capacity._vp._parametric
                    var = pyo.Var(initialize = 0.5*opt_bounds[1], bounds = (opt_bounds[0], opt_bounds[1]))
                    setattr(self._m, f'{comp.name}', var)
                elif mode == 'SweepValues': #TODO Add capability to handle sweepvalues, maybe multiple pyo.Params?
                    raise IOError('MOPED does not currently support sweep values option')
                elif mode == 'FixedValue':
                    value = prod._capacity._vp._parametric
                    param = pyo.Param(initialize = value)
                    setattr(self._m, f'{comp.name}', param)
                component_meta[comp.name]['Demands'] = resource
                component_meta[comp.name]['Consumes'] = {}
                component_meta[comp.name]['Dispatch'] = dem._dispatchable
                component_meta[comp.name]['Pyomo'] = getattr(self._m, f'{comp.name}')



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

    def getTargetParams(self, target = 'all'):
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
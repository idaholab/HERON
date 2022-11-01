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
import pandas as pd
import matplotlib.pyplot as plt

from HERON.src import _utils as hutils
from HERON.src.base import Base
try:
  import ravenframework
except ModuleNotFoundError:
  path_to_raven = hutils.get_raven_loc()
  sys.path.append(os.path.abspath(os.path.join(path_to_raven, 'plugins')))
  sys.path.append(path_to_raven)
from TEAL.src import main as RunCashFlow
from TEAL.src import CashFlows
from ravenframework.ROMExternal import ROMLoader
from ravenframework.MessageHandler import MessageHandler

class MOPED(Base):
  def __init__(self):
    """
      Construct.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._components = []                 # List of components objects from heron input
    self._sources = []                    # List of sources objects from heron input
    self._case = None                     # Case object that contains the case parameters
    self._econ_settings = None            # TEAL global settings used for building cashflows
    self._m = None                        # Pyomo model to be solved
    self._producers = []                  # List of pyomo var/params of producing components
    self._eval_mode = 'clustered'         # (full or clustered) clustered is better for testing and speed, full gives a more realistic NPV result
    self._yearly_hours = 24 * 365         # Number of hours in a year to handle dispatch, based on clustering
    self._component_meta = {}             # Primary data structure for MOPED, organizes important information for problem construction
    self._cf_meta = {}                    # Secondary data structure for MOPED, contains cashflow info
    self._resources = []                  # List of resources used in this analysis
    self._solver = SolverFactory('ipopt') # Solver for optimization solve, default is 'ipopt', TODO allow user to specify
    self._cf_components = []              # List of TEAL.Components objects generated for analysis
    self._dispatch = []                   # List of pyomo vars/params for each realization and year
    self._multiplicity_meta = {}          # Dictionary of analysis years, clusters, and associated multiplicity
    self._plot = False                    # Boolean to determine if a dispatch plot is made for the analysis (defaults to false)

    self.messageHandler = MessageHandler()

  def buildActivity(self):
    """
      Generate active list that is necessary for building TEAL settings object
      @ In, None
      @ Out, activity, list, associates component with cashflow types ([ngcc|'Cap', ngcc|'Hourly'])
    """
    activity = []
    for comp in self._components:
      # TODO Does this need to be expanded on?
      for cf in comp._economics._cash_flows:
        if cf._type == 'one-time':
          activity.append(f'{comp.name}|Capex')
        elif cf._type == 'repeating':
          if cf._period == 'year':
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
      for prod in comp._produces:  # NOTE Cannot handle components that produce multiple things
        resource = prod._capacity_var
        mode = prod._capacity.type
        self.setCapacityMeta(mode, resource, comp, prod, kind='produces')
      for dem in comp._demands:  # NOTE Cannot handle components that demand multiple things
        resource = dem._capacity_var
        mode = dem._capacity.type
        self.setCapacityMeta(mode, resource, comp, dem, kind='demands')
      for store in comp._stores:  # NOTE Cannot handle components that store multiple things
        resource = store._capacity_var
        mode = store._capacity.type
        self.setCapacityMeta(mode, resource, comp, store, kind='stores')

  def buildMultiplicityMeta(self):
    """
      Loads source structure and builds appropriate multiplicity data
      @ In, None
      @ Out, None
    """
    structure = hutils.get_synthhist_structure(self._sources[0]._target_file)
    cluster_years = sorted(structure['clusters'])
    for i in range(len(cluster_years)):
      self._multiplicity_meta[i+1] = {}
      # Necessary to still allow full eval mode
      if self._eval_mode == 'full':
        self._multiplicity_meta[i+1][0] = 1
        continue
      cluster_data = structure['clusters'][cluster_years[i]]
      for cluster_info in cluster_data:
        self._multiplicity_meta[i+1][cluster_info['id']] = len(cluster_info['represents'])
    self._multiplicity_meta['Index Map'] = '[Year][Cluster][Multiplicity]'

  def loadSyntheticHistory(self, signal, multiplier):
    """
      Loads synthetic history for a specified signal,
      also sets yearly hours and pyomo indexing sets
      @ In, signal, string, name of signal to sample
      @ In, multiplier, int/float, value to multiply synthetic history evaluations by
      @ Out, synthetic_data, dict, contains data from evaluated ROM
    """
    # NOTE self._sources[0]._var_names are the user assigned signal names in DataGenerators
    if signal not in self._sources[0]._var_names:
      raise IOError('The requested signal name is not available'
                    'from the synthetic history, check DataGenerators node in input')
    runner = ROMLoader(self._sources[0]._target_file)
    from ravenframework.utils import xmlUtils
    inp = {'scaling': [1]}
    # TODO expand to change other pickledROM settings withing this method
    nodes = []
    node = xmlUtils.newNode('ROM', attrib={'name': 'SyntheticHistory', 'subType': 'pickledRom'})
    node.append(xmlUtils.newNode('clusterEvalMode', text=self._eval_mode))
    nodes.append(node)
    runner.setAdditionalParams(nodes)
    synthetic_data = {}
    for real in range(self._case._num_samples):
      self.raiseAMessage(f'Loading synthetic history for signal: {signal}')
      name = f'Realization_{real + 1}'
      current_realization = runner.evaluate(inp)[0]
      # applying mult to each realization is easier than iteration through dict object later
      current_realization[signal] *= multiplier
      if self._eval_mode == 'full':
        # reshape so that a filler cluster index is made
        current_realization[signal] = np.expand_dims(current_realization[signal], axis=1)
      synthetic_data[name] = current_realization[signal]
    cluster_count = synthetic_data['Realization_1'].shape[1]
    hour_count = synthetic_data['Realization_1'].shape[2]
    self._m.c = pyo.Set(initialize=np.arange(cluster_count))
    if self._eval_mode not in ['clustered', 'full']:
      raise IOError('Improper ROM evaluation mode detected, try "clustered" or "full".')
    # How many dispatch points we will have for each year
    self._yearly_hours = hour_count * cluster_count
    self._m.t = pyo.Set(initialize=np.arange(hour_count))
    return synthetic_data

  def loadStaticHistory(self, signal, multiplier):
    """
      Loads static history for a specified signal,
      also sets yearly hours and pyomo indexing sets
      @ In, signal, string, name of signal to sample
      @ In, multiplier, int/float, value to multiply synthetic history evaluations by
      @ Out, synthetic_data, dict, contains data from evaluated ROM
    """
    synthetic_data = {}
    # TODO: this is being implemented in HERD but makes sense to also add in MOPED
    raise IOError('Static histories not yet implemented in MOPED.')
    return synthetic_data

  def setCapacityMeta(self, mode, resource, comp, element, kind='produces'):
    """
      Checks the capacity type, dispatch type, and resources involved for each component
      to build component_meta
      @ In, mode, string, type of capacity definition for component
      @ In, resource, string, resource produced or demanded
      @ In, comp, HERON component
      @ In, element, HERON produces/demands node
      @ In, kind, string, describes comp type, stores,produces,demands
      @ Out, None
    """
    # Multiplier plays important role in capacity node, especially for VRE's
    capacity_mult = element._capacity._multiplier
    # For MOPED we treat all capacities and dispatches as positive values
    if capacity_mult is None:
      capacity_mult = 1
    elif capacity_mult < 0:
      capacity_mult *= -1
    self._component_meta[comp.name]['Capacity Resource'] = resource
    self._component_meta[comp.name]['Stores'] = None
    self._component_meta[comp.name]['Produces'] = None
    self._component_meta[comp.name]['Demands'] = None
    # Organizing important aspects of problem for later access
    # TODO considering lists of produce and demand
    if kind == 'produces':
      self._component_meta[comp.name]['Produces'] = element._produces[0]
    elif kind == 'demands':
      self._component_meta[comp.name]['Demands'] = element._demands[0]
    elif kind == 'stores':
      self._component_meta[comp.name]['Stores'] = element._stores
      self._component_meta[comp.name]['Initial Value'] = element._initial_stored.get_value()
      self._component_meta[comp.name]['SRTE'] = element.get_sqrt_RTE()
    self._component_meta[comp.name]['Consumes'] = None
    self._component_meta[comp.name]['Dispatch'] = element._dispatchable
    # Different possible capacity value definitions for a component
    if mode == 'OptBounds':
      self.raiseADebug(f'Building pyomo capacity variable for '
                       f'{comp.name}')
      opt_bounds = element._capacity._vp._parametric
      # Considering user inputs for default heron sign convention
      if opt_bounds[1] < 1:
        opt_bounds *= -1
      opt_bounds *= capacity_mult
      # This is a capacity we make a decision on
      var = pyo.Var(initialize=0.5 * opt_bounds[1], bounds=(opt_bounds[0], opt_bounds[1]))
      setattr(self._m, f'{comp.name}', var)
    elif mode == 'SweepValues':  # TODO Add capability to handle sweepvalues, maybe multiple pyo.Params?
      raise IOError('MOPED does not currently support sweep values option')
    elif mode == 'FixedValue':
      self.raiseADebug(f'Building pyomo capacity parameter for '
                       f'{comp.name}')
      # Params represent constant value components of the problem
      value = element._capacity._vp._parametric
      # Considering user inputs for default heron sign convention
      if value < 1:
        value *= -1
      value *= capacity_mult
      param = pyo.Param(initialize=value)
      setattr(self._m, f'{comp.name}', param)
    elif mode == 'SyntheticHistory':
      self.raiseADebug(f'Building capacity with synthetic histories for '
                       f'{comp.name}')
      # This method runs external ROM loader and defines some pyomo sets
      capacity = self.loadSyntheticHistory(element._capacity._vp._var_name, capacity_mult)
      # TODO how to better handle capacities based on Synth Histories
      self._component_meta[comp.name]['Capacity'] = capacity
    elif mode == 'StaticHistory':
      self.raiseADebug(f'Building capacity with static histories for '
                       f'{comp.name}')
      # This method runs external ROM loader and defines some pyomo sets
      capacity = self.loadStaticHistory(element._capacity._vp._var_name, capacity_mult)
      # TODO how to better handle capacities based on Synth Histories
      self._component_meta[comp.name]['Capacity'] = capacity
    if mode != 'SyntheticHistory':
      # TODO smarter way to do this check?
      self._component_meta[comp.name]['Capacity'] = getattr(self._m, f'{comp.name}')
    if kind == 'produces':
      # TODO should we handle transfer functions here?
      for con in element._consumes:
        transfer_values = element.get_transfer().get_coefficients()
        self._component_meta[comp.name]['Consumes'] = con
        self._component_meta[comp.name]['Transfer'] = abs(transfer_values[con]) / abs(transfer_values[element._produces[0]])

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
        # This is used later in cashflow object generation, can be unique to each cf a comp has
        params = {'tax':cf._taxable,
                  'inflation':cf._inflation,
                  'mult_target':cf._mult_target,
                  'reference':cf._reference.get_value(),
                  'X':cf._scale.get_value(),
                  }
        if params['inflation'] == 'none':
          params['inflation'] = None
        multiplier = cf._driver._multiplier
        driver_type = cf._driver.type
        # Default mult should be 1
        if multiplier == None:
          multiplier = 1
        # This corrects sign for MOPED from user inputs for demanding cashflows
        # Allows MOPED and default HERON to follow same sign conventions for inputs
        if len(comp._demands) > 0:
          multiplier *= -1
        # Using reference prices for cashflows, considering uncertain market prices
        if cf._alpha.type == 'SyntheticHistory':
          signal = cf._alpha._vp._var_name
          alpha = self.loadSyntheticHistory(signal, multiplier)
        elif cf._alpha.type == 'StaticHistory':
          signal = cf._alpha._vp._var_name
          alpha = self.loadStaticHistory(signal, multiplier)
        else:
          alpha = cf._alpha._vp._parametric * multiplier
        if cf._type == 'one-time':
          # TODO consider other driver types
          if driver_type == 'FixedValue':
            self._cf_meta[comp.name]['Capex Driver'] = cf._driver._vp._parametric
          else:
            self._cf_meta[comp.name]['Capex Driver'] = None
          self._cf_meta[comp.name]['Capex'] = alpha
          self._cf_meta[comp.name]['Capex Params'] = params
          # Necessary if capex has depreciation and amortization
          self._cf_meta[comp.name]['Deprec'] = cf._depreciate
        elif cf._type == 'repeating':
          if cf._period == 'year':
            if driver_type == 'FixedValue':
              self._cf_meta[comp.name]['Yearly Driver'] = cf._driver._vp._parametric
            else:
              self._cf_meta[comp.name]['Yearly Driver'] = None
            self._cf_meta[comp.name]['Yearly'] = alpha
            self._cf_meta[comp.name]['Yearly Params'] = params
            continue
          if driver_type == 'FixedValue':
            self._cf_meta[comp.name]['Dispatch Driver'] = cf._driver._vp._parametric
          else:
            self._cf_meta[comp.name]['Dispatch Driver'] = None
          self._cf_meta[comp.name]['Dispatching'] = alpha
          self._cf_meta[comp.name]['Dispatching Params'] = params

  def createCapex(self, comp, alpha, capacity, unique_params):
    """
      Builds capex TEAL cashflow for a given component
      @ In, comp, TEAL component object
      @ In, alpha, float, reference price for capex cost
      @ In, capacity, pyomo var, size of the ocmponent that drives the cost
      @ In, unique_params, dict, settings for inflation, tax, and mult for cf
      @ Out, cf, TEAL cashflow
    """
    life = comp.getLifetime()
    cf = CashFlows.Capex()
    cf.name = 'Capex'
    cf.initParams(life)
    cfParams = {'name': 'Capex',
                'alpha': alpha,
                'driver': capacity,
                'reference': unique_params['reference'],
                'X': unique_params['X'],
                'mult_target': unique_params['mult_target'],
                }
    cf.setParams(cfParams)
    return cf

  def createRecurringYearly(self, comp, alpha, driver, unique_params):
    """
      Constructs the parameters for capital expenditures
      @ In, comp, TEAL.src.CashFlows.Component, main structure to add component cash flows
      @ In, alpha, float, yearly price to populate
      @ In, driver, pyomo.core.base.var.ScalarVar, quantity sold to populate
      @ In, unique_params, dict, settings for inflation, tax, and mult for cf
      @ Out, cf, TEAL.src.CashFlows.Component, cashflow sale for the recurring yearly
    """
    # Necessary to make life integer valued for numpy
    life = int(self._case._global_econ['ProjectTime'])
    cf = CashFlows.Recurring()
    cfParams = {'name': 'Yearly',
                'X': unique_params['X'],
                'mult_target': unique_params['mult_target'],
                }
    cf.setParams(cfParams)
    # 0 for first year (build year) -> TODO couldn't this be automatic?
    alphas = np.ones(life + 1, dtype=object) * alpha
    drivers = np.ones(life + 1, dtype=object) * driver
    alphas[0] = 0
    drivers[0] = 0
    # construct annual summary cashflows
    cf.computeYearlyCashflow(alphas, drivers)
    return cf

  def createRecurringHourly(self, comp, alpha, driver, real, unique_params):
    """
      Generates recurring hourly cashflows, mostly for dispatch and sales
      @ In, comp, TEAL component
      @ In, alpha, float/np.array, reference price of sale
      @ In, driver, numpy array of pyomo.var.values that drive cost
      @ In, real, int, current realization number
      @ In, unique_params, dict, settings for inflation, tax, and mult for cf
      @ Out, cf, TEAL cashflow
    """
    # Necessary to make integer for numpy arrays
    life = int(self._case._global_econ['ProjectTime'])
    cf = CashFlows.Recurring()
    cfParams = {'name': f'Dispatching_{real+1}',
                'X': unique_params['X'],
                'mult_target': unique_params['mult_target'],
                }
    cf.setParams(cfParams)
    cf.initParams(life, pyomoVar=True)
    # Necessary to shift year index by one since no recurring cashflows on first build year
    for year in range(life + 1):
      # Alpha can be a fixed single value price or an array of prices for each timestep
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
        if prod._capacity_var not in self._resources:
          self._resources.append(prod._capacity_var)
          # TODO add for consuming components
      for dem in comp._demands:
        resource = dem._capacity_var
        if resource not in self._resources:
            self._resources.append(resource)

  def buildMultiplicityVariables(self):
    """
      Generates pyomo params for applying multiplicity to dispatch vars/params
      @ In, None
      @ Out, None
    """
    if self._eval_mode == 'clustered':
      self.raiseADebug('Building multiplicity vector for clustered ROM evaluation...')
    else:
      self.raiseADebug('Building multiplicity filler for full ROM evaluation...')
    project_life = int(self._case._global_econ['ProjectTime'])
    for year in range(project_life):
      # Multiplicity used to scaled dispatches based on cluster and year
      mult = pyo.Param(self._m.c, self._m.t,
                          initialize=lambda m, c, t: self._multiplicity_meta[year+1][c],
                          domain=pyo.NonNegativeReals
                          )
      setattr(self._m, f'multiplicity_{year+1}',mult)

  def buildDispatchVariables(self, comp):
    """
      Generates dispatch vars and value arrays to build components
      @ In, comp, HERON component
      @ Out, capacity, np.array/pyomo.var, capacity variable for the component
      @ Out, template_array, np.array, array of pyo.values used for TEAL cfs
    """
    # NOTE Assumes that all components will remain functional for project life
    project_life = int(self._case._global_econ['ProjectTime'])
    # Necessary to make year index one larger than project life so that year zero
    # Can be empty for recurring cashflows
    template_array = np.zeros((self._case._num_samples, project_life + 1, self._yearly_hours), dtype=object)
    capacity = self._component_meta[comp.name]['Capacity']
    dispatch_type = self._component_meta[comp.name]['Dispatch']
    # What to have user be able to define consuming components capacity in terms of either resource
    # This allows dispatch to be in terms of producing resource and capacity be in terms of consumption resource
    # Only applies to fixed dispatch here due to pyomo construction
    if self._component_meta[comp.name]['Consumes'] is not None:
      if self._component_meta[comp.name]['Consumes'] == self._component_meta[comp.name]['Capacity Resource']:
        reverse_transfer = 1 / self._component_meta[comp.name]['Transfer']
    else:
      reverse_transfer = 1
    # Checking for type of capacity is necessary to build dispatch variable
    self._m.dummy = pyo.Var()
    self._m.placeholder = pyo.Param()
    dummy_type = type(self._m.dummy)
    placeholder_type = type(self._m.placeholder)
    self.raiseADebug(f'Preparing dispatch container for {comp.name}...')
    for real in range(self._case._num_samples):
      for year in range(project_life):
        mult = getattr(self._m,f'multiplicity_{year+1}')
        # TODO account for other variations of component settings, specifically if dispatchable
        if isinstance(capacity, (dummy_type, placeholder_type)):
          # Currently independent and dependent are interchangable
          if dispatch_type in ['independent', 'dependent']:
            var = pyo.Var(self._m.c, self._m.t,
                          initialize=lambda m, c, t: 0,
                          domain=pyo.NonNegativeReals
                          )
            setattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}', var)
            # Shifting index such that year 0 remains 0
            # Weighting each dispatch by the number of realizations (equal weight for each realization)
            # This corrects the NPV value
            template_array[real, year + 1, :] = (1 / self._case._num_samples) * np.array(list(var.values())) * np.array(list(mult.values()))
          elif dispatch_type == 'fixed':
            param = pyo.Var(self._m.c, self._m.t,
                            initialize=lambda m, c, t: capacity.value,
                            domain=pyo.NonNegativeReals,)
            setattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}', param)
            con = pyo.Constraint(self._m.c, self._m.t, expr=lambda m, c, t: param[(c, t)] == reverse_transfer*capacity)
            setattr(self._m, f'{comp.name}_fixed_{real+1}_{year+1}', con)
            template_array[real, year + 1, :] = (1 / self._case._num_samples) * np.array(list(param.values())) * np.array(list(mult.values()))
        else:
          if dispatch_type in ['independent', 'dependent']:
            var = pyo.Var(self._m.c, self._m.t,
                          initialize=lambda m, c, t: 0,
                          domain=pyo.NonNegativeReals,
                          bounds=lambda m, c, t: (0, capacity[f'Realization_{real+1}'][year, c, t])
                          )
            setattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}', var)
            template_array[real, year + 1, :] = (1 / self._case._num_samples) * np.array(list(var.values())) * np.array(list(mult.values()))
          elif dispatch_type == 'fixed':
            param = pyo.Param(self._m.c, self._m.t,
                              initialize=lambda m, c, t: reverse_transfer*capacity[f'Realization_{real+1}'][year, c, t]
                              )
            setattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}', param)
            template_array[real, year + 1, :] = (1 / self._case._num_samples) * np.array(list(param.values())) * np.array(list(mult.values()))
    return capacity, template_array

  def buildConsumptionVariables(self, comp):
    """
      Builds consumption pyomo variables that are dependent on the output of the same components dispatch
      @ In, comp, HERON component object
      @ Out, None
    """
    # NOTE Assumes that all components will remain functional for project life
    project_life = int(self._case._global_econ['ProjectTime'])
    transfer = self._component_meta[comp.name]['Transfer']
    for real in range(self._case._num_samples):
      for year in range(project_life):
        dispatch = getattr(self._m,f'{comp.name}_dispatch_{real+1}_{year+1}')
        var = pyo.Var(self._m.c, self._m.t,
                      initialize=lambda m, c, t: 0,
                      domain=pyo.NonNegativeReals)
        setattr(self._m,f'{comp.name}_consume_{real+1}_{year+1}',var)
        con = pyo.Constraint(self._m.c,self._m.t,
                             rule=lambda m, c, t: var[(c,t)] == transfer*dispatch[(c,t)])
        setattr(self._m,f'{comp.name}_consumption_limit_{real+1}_{year+1}',con)

  def buildStorageVariables(self, comp):
    """
      Builds storage dispatch(charge/discharge), level, and dependencies in pyomo
      @ In, comp, HERON component object
      @ Out, capacity, np.array/pyomo.var, capacity variable for the component
      @ Out, template_array, np.array, array of pyo.values used for TEAL cfs
    """
    self.raiseADebug(f'Preparing storage variables for {comp.name}')
    # NOTE Assumes that all components will remain functional for project life
    project_life = int(self._case._global_econ['ProjectTime'])
    # Necessary to make year index one larger than project life so that year zero
    # Can be empty for recurring cashflows
    template_array = np.zeros((self._case._num_samples, project_life + 1, self._yearly_hours), dtype=object)
    capacity = self._component_meta[comp.name]['Capacity']
    # NOTE we assume independent for all storage components
    dispatch_type = self._component_meta[comp.name]['Dispatch']
    initial_value = self._component_meta[comp.name]['Initial Value']
    trip_efficiency = self._component_meta[comp.name]['SRTE']
    # TODO how to dynamically generate the time-step value?
    dt = self._m.t[2] - self._m.t[1]
    cluster_end = self._m.t[-1]
    for real in range(self._case._num_samples):
      for year in range(project_life):
        mult = getattr(self._m,f'multiplicity_{year+1}')
        # battery needs to track level, charging, and discharging
        level = pyo.Var(self._m.c, self._m.t,
                        domain=pyo.NonNegativeReals)
        setattr(self._m,f'{comp.name}_level_{real+1}_{year+1}',level)
        level_upper = pyo.Constraint(self._m.c, self._m.t,
                                     rule=lambda m, c, t: level[(c,t)] <= capacity)
        setattr(self._m,f'{comp.name}_level_upper_{real+1}_{year+1}',level_upper)
        charge = pyo.Var(self._m.c, self._m.t,
                         domain=pyo.NonNegativeReals)
        setattr(self._m,f'{comp.name}_charge_{real+1}_{year+1}',charge)
        discharge = pyo.Var(self._m.c, self._m.t,
                         domain=pyo.NonNegativeReals)
        setattr(self._m,f'{comp.name}_discharge_{real+1}_{year+1}',discharge)
        discharge_limit = pyo.Constraint(self._m.c, self._m.t,
                                         rule=lambda m, c, t:
                                         discharge[(c,t)] <= trip_efficiency*self.getPreviousIndex(comp.name, level, real+1, year+1, c, t, initial_value))
        setattr(self._m,f'{comp.name}_discharge_limit_{real+1}_{year+1}',discharge_limit)
        # level is time dependent and requires propagation via constraints
        level_propagation = pyo.Constraint(self._m.c, self._m.t,
                                           rule=lambda m, c, t:
                                           level[(c,t)] == self.getPreviousIndex(comp.name, level, real+1, year+1, c, t, initial_value) +
                                           dt*(trip_efficiency*charge[(c,t)] - (1/trip_efficiency)*discharge[(c,t)]))
        setattr(self._m,f'{comp.name}_level_propagation_{real+1}_{year+1}',level_propagation)
        # Storage set points should enforce shorter time horizons for storage decisions
        level_point_set_lower = pyo.Constraint(self._m.c,
                                         rule=lambda m, c: level[(c,0)] == initial_value)
        setattr(self._m,f'{comp.name}_level_setpoint_lower_{real+1}_{year+1}',level_point_set_lower)
        level_point_set_upper = pyo.Constraint(self._m.c,
                                         rule=lambda m, c: level[(c,cluster_end)] == initial_value)
        setattr(self._m,f'{comp.name}_level_setpoint_upper_{real+1}_{year+1}',level_point_set_upper)
        # TODO currently only considering costs associated with discharging the storage, however charging and level should be considered
        # This will involve handling a separate template_array as a driver for a separate TEAL cashflow
        template_array[real, year+1, :] = (1 / self._case._num_samples) * np.array(list(discharge.values())) * np.array(list(mult.values()))
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
    params = {'name': comp.name}
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
        capex_params = cf_meta['Capex Params']
        capex_driver = cf_meta['Capex Driver']
        if capex_driver is None:
          capex = self.createCapex(component, value, capacity, capex_params)
        else:
          capex = self.createCapex(component, value, capex_driver, capex_params)
        cfs.append(capex)
        depreciation = cf_meta['Deprec']
        if depreciation is not None:
          capex.setAmortization('MACRS', depreciation)
          amorts = component._createDepreciation(capex)
          cfs.extend(amorts)
      # Necessary to avoid error message from expected inputs
      elif cf in ['Deprec', 'Capex Driver', 'Yearly Driver', 'Dispatch Driver', 'Capex Params', 'Yearly Params', 'Dispatching Params']:
        continue
      elif cf == 'Yearly':
        self.raiseADebug(f'Generating Yearly OM cashflow for {comp.name}')
        yearly_params = cf_meta['Yearly Params']
        yearly_driver = cf_meta['Yearly Driver']
        if yearly_driver is None:
          yearly = self.createRecurringYearly(component, value, capacity, yearly_params)
        else:
          yearly = self.createRecurringYearly(component, value, yearly_driver, yearly_params)
        cfs.append(yearly)
      elif cf == 'Dispatching':
        # Here value can be a np.array as well for ARMA grid pricing
        self.raiseADebug(f'Generating dispatch OM cashflow for {comp.name}')
        dispatching_params = cf_meta['Dispatching Params']
        dispatch_driver = cf_meta['Dispatch Driver']
        if dispatch_driver is None:
          if isinstance(value, dict):
            for real in range(self._case._num_samples):
              alpha_realization = self.reshapeAlpha(value)
              var_om = self.createRecurringHourly(component, alpha_realization[real, :, :], dispatch[real, :, :], real, dispatching_params)
              cfs.append(var_om)
          else:
            # Necessary to create a unique cash flow for each dispatch realization
            for real in range(self._case._num_samples):
              var_om = self.createRecurringHourly(component, value, dispatch[real, :, :], real, dispatching_params)
              cfs.append(var_om)
        else:
          raise IOError('MOPED does not currently handle non activity drivers for dispatch recurring cashflows')
      else:
        raise IOError(f'Unexpected cashflow type received: {cf}')
    component.addCashflows(cfs)
    return component

  def reshapeAlpha(self, alpha):
    """
      Reshapes synthetic history reference prices to match array shape of driver arrays
      @ In, alpha, dict, dictionary of numpy arrays
      @ Out, reshaped_alpha, numpy array, same data in new shape
    """
    project_life = int(self._case._global_econ['ProjectTime'])
    # plus 1 to year term to allow for 0 recurring costs during build year
    reshaped_alpha = np.zeros((self._case._num_samples,project_life+1,self._yearly_hours))
    for real in range(self._case._num_samples):
      # it necessary to have alpha be [real,year,hour] instead of [real,year,cluster,hour]
      realized_alpha = np.hstack([alpha[f'Realization_{real+1}'][:,i,:] for i in range(alpha[f'Realization_{real+1}'].shape[1])])
      reshaped_alpha[real,1:,:] = realized_alpha
    # TODO effective way of checking to see if reshape was successful?
    return reshaped_alpha

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
    consumed = 0
    charged = 0
    discharged = 0
    # Necessary to check all components involved in the analysis
    for comp in self._components:
      comp_meta = self._component_meta[comp.name]
      # Conservation constrains the dispatch decisions
      if comp_meta['Stores'] is not None:
        charge_value = getattr(self._m, f'{comp.name}_charge_{real + 1}_{year + 1}')
        discharge_value = getattr(self._m, f'{comp.name}_discharge_{real + 1}_{year + 1}')
      else:
        dispatch_value = getattr(self._m, f'{comp.name}_dispatch_{real + 1}_{year + 1}')
      if comp_meta['Consumes'] is not None:
        consumption_value = getattr(self._m,f'{comp.name}_consume_{real+1}_{year+1}')
      for key, value in comp_meta.items():
        if key == 'Produces' and value == resource:
          produced += dispatch_value[(c,t)]
        elif key == 'Demands' and value == resource:
          demanded += dispatch_value[(c,t)]
        elif key == 'Consumes' and value == resource:
          consumed += consumption_value[(c,t)]
        elif key == 'Stores' and value == resource:
          charged += charge_value[(c,t)]
          discharged += discharge_value[(c,t)]
        # TODO consider consumption and incorrect input information
    return produced + discharged == demanded + consumed + charged

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
    # What to have user be able to define consuming components capacity in terms of either resource
    # This allows dispatch to be in terms of producing resource and capacity be in terms of consumption resource
    if self._component_meta[comp.name]['Consumes'] is not None:
      if self._component_meta[comp.name]['Consumes'] == self._component_meta[comp.name]['Capacity Resource']:
        reverse_transfer = 1 / self._component_meta[comp.name]['Transfer']
    else:
      reverse_transfer = 1
      # This is allows for the capacity to be an upper bound and decision variable
    upper_bound = reverse_transfer*getattr(self._m, f'{comp.name}')
    dispatch_value = getattr(self._m, f'{comp.name}_dispatch_{real+1}_{year+1}')
    return dispatch_value[(c, t)] <= upper_bound

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
                               rule=partial(self.conserveResource, resource, real, year))
          setattr(self._m, f'{resource}_con_{real+1}_{year+1}', con)
        # Bounding constraints on dispatches
        for comp in self._components:
          # storage has constraints build elsewhere see (buildStorageComponents)
          if self._component_meta[comp.name]['Stores'] is not None:
            continue
          capacity = self._component_meta[comp.name]['Capacity']
          if isinstance(capacity, (dummy_type, placeholder_type)):
            con = pyo.Constraint(self._m.c, self._m.t,
                                 rule=partial(self.upper, comp, real, year))
            setattr(self._m, f'{comp.name}_upper_{real+1}_{year+1}', con)

  def solveAndDisplay(self):
    """
      Presents results of the optimization run
      @ In, None
      @ Out, None
    """
    columns = []
    values = []
    # Results provide run times and optimizer final status
    results = self._solver.solve(self._m)
    self.raiseAMessage(f'Optimizer has finished running, here are the results\n{results}')
    for comp in self._components:
      # Not all components will have a pyomo variable
      try:
        comp_print = getattr(self._m, f'{comp.name}')
        self.raiseAMessage(f'Here is the optimized capacity for {comp.name}')
        columns.append(f'{comp.name} Capacity')
        values.append(comp_print.value)
        comp_print.pprint()
      except:
        self.raiseAMessage(f'{comp.name} does not have a standard capacity')
    NPV = pyo.value(self._m.NPV)
    self.raiseAMessage(f"The final NPV is: {NPV}")
    columns.append('Expected NPV')
    values.append(NPV)
    output_data = pd.DataFrame([values], columns=columns)
    output_data.to_csv('opt_solution.csv')

  def dispatchPlot(self, real=1, year=1, cluster=0):
    """
      Plots the dispatch behavior for a given realization, year, and cluster
      @ In, real, int, realization to plot (defaults to 1)
      @ In, year, int, year to plot (defaults to 1)
      @ In, cluster, int, cluster to plot (defaults to 1)
      @ Out, None
    """
    self.raiseAMessage(f'Generating resource dispatch plots for {self._case.name}')
    time = np.array(self._m.t)
    for res in self._resources:
      plot_colors = ['green','red','blue','orange','teal','violet','brown','black','yellow']
      plt.figure(figsize=(2,1))
      main = plt.subplot(111)
      main.set_xlabel('Time')
      main.set_ylabel(f'{res} (Dispatched)')
      main.set_title(f'{res} Dispatch (Realization: {real} Year: {year} Cluster: {cluster})')
      main.set_xlim((0,time[-1]))
      for comp in self._components:
        if self._component_meta[comp.name]['Produces'] == res:
          plot_dispatch = np.zeros(len(self._m.t))
          dispatch = getattr(self._m,f'{comp.name}_dispatch_{real}_{year}')
          for t in self._m.t:
            plot_dispatch[t] = pyo.value(dispatch[(cluster,t)])
          label = f'{comp.name} Production'
          main.plot(time,plot_dispatch,label=label,color=plot_colors[0])
          plot_colors.pop(0)
        elif self._component_meta[comp.name]['Demands'] == res:
          plot_dispatch = np.zeros(len(self._m.t))
          dispatch = getattr(self._m,f'{comp.name}_dispatch_{real}_{year}')
          for t in self._m.t:
            plot_dispatch[t] = -1*pyo.value(dispatch[(cluster,t)])
          if self._component_meta[comp.name]['Dispatch'] =='fixed':
            label = f'{comp.name} Demand'
          else:
            label = f'{comp.name} Sales'
          main.plot(time,plot_dispatch,label=label,color=plot_colors[0])
          plot_colors.pop(0)
        elif self._component_meta[comp.name]['Consumes'] == res:
          plot_dispatch = np.zeros(len(self._m.t))
          dispatch = getattr(self._m,f'{comp.name}_consume_{real}_{year}')
          for t in self._m.t:
            plot_dispatch[t] = -1*pyo.value(dispatch[(cluster,t)])
          label = f'{comp.name} Consumption'
          main.plot(time,plot_dispatch,label=label,color=plot_colors[0])
          plot_colors.pop(0)
        elif self._component_meta[comp.name]['Stores'] == res:
          plot_level = np.zeros(len(self._m.t))
          plot_charge = np.zeros(len(self._m.t))
          plot_discharge = np.zeros(len(self._m.t))
          level = getattr(self._m,f'{comp.name}_level_{real}_{year}')
          charge = getattr(self._m,f'{comp.name}_charge_{real}_{year}')
          discharge = getattr(self._m,f'{comp.name}_discharge_{real}_{year}')
          for t in self._m.t:
            plot_level[t] = pyo.value(level[(cluster,t)])
            plot_charge[t] = -1*pyo.value(charge[(cluster,t)])
            plot_discharge[t] = pyo.value(discharge[(cluster,t)])
          label = f'{comp.name} Charging'
          main.plot(time,plot_charge,label=label,color=plot_colors[0],marker='x',ls='--')
          label = f'{comp.name} Discharging'
          main.plot(time,plot_discharge,label=label,color=plot_colors[0],marker='o',ls='--')
          sec_axis = main.twinx()
          sec_axis.set_ylabel(f'{res} (Stored)')
          cap = pyo.value(getattr(self._m,f'{comp.name}'))
          sec_axis.set_ylim((0,cap+1))
          label = f'{comp.name} Level'
          sec_axis.plot(time,plot_level,label=label,color=plot_colors[0],ls='-.')
          plot_colors.pop(0)
      leg_main = main.legend(loc='best')
      leg_main.set_title('Dispatch')
      leg_sec = sec_axis.legend(loc='best')
      leg_sec.set_title('Storage')
      plt.show()

  # ===========================
  # MAIN WORKFLOW
  # ===========================
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
    self.buildMultiplicityMeta()
    self.collectResources()
    self.buildMultiplicityVariables()
    # Each component will have dispatch and cashflow associated
    for comp in self._components:
      # Storage components have their own unique set of pyomo variables
      if self._component_meta[comp.name]['Stores'] is None:
        capacity, dispatch = self.buildDispatchVariables(comp)
      else:
        capacity, dispatch = self.buildStorageVariables(comp)
      cf_comp = self.createCashflowComponent(comp, capacity, dispatch)
      if self._component_meta[comp.name]['Consumes'] is not None:
        self.buildConsumptionVariables(comp)
      self._cf_components.append(cf_comp)
    self.raiseAMessage(f'Building pyomo cash flow expression for {self._case.name}')
    # TEAL is our cost function generator here
    metrics = RunCashFlow.run(self._econ_settings, self._cf_components, {}, pyomoVar=True)
    self._m.NPV = pyo.Objective(expr=metrics['NPV'], sense=pyo.maximize)
    # Constraints need to be built for conservation and bounds of dispatch
    self.buildConstraints()
    # NOTE this currently displays just optimizer info and capacities and cost funtion
    # TODO does this need to present information about dispatches, how to do this?
    self.raiseAMessage(f'Running Optimizer...')
    self.solveAndDisplay()
    # TODO provide way for user to turn plotting on and off, defaults to off
    if self._plot:
      self.dispatchPlot()

  # ===========================
  # UTILITIES
  # ===========================
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
                                    'suppressErrs': False, })
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
      @ In, target, string, param to retrieve, defaults to 'all'
      @ Out, case, Cases.Case object
      @ Out, components, list of Components.Component objects
      @ Out, sources, list of Placeholder objects
    """
    case = self._case
    components = self._components
    sources = self._sources
    # TODO Expand this method to include all attributes that are useful to retrieve
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

  def getPreviousIndex(self, comp_name, variable, real, year, c, t, initial_value):
    """
      Given an indexed variable, returns the value of the same variable, but of the previous index
      This is specifically useful for the development of battery level constraints
      @ In, comp_name, string, name of component that variable belongs to
      @ In, variable, pyomo var object
      @ In, real, int, realization of the variable
      @ In, year, int, year of the variable
      @ In, c, int, current cluster index
      @ In, t, int, current time index
      @ In, initial_value, float, initial value of var prior to analysis start
      @ Out, previous_index, indexed pyomo var object
    """
    # Need to know length of cluster and time indexes for getting previous data
    time = len(self._m.t)
    cluster = len(self._m.c)
    if t == 0:
      if c == 0:
        if year == 1:
          return initial_value
        else:
          previous_level = getattr(self._m,f'{comp_name}_level_{real}_{year}')
          # Indexing pyomo vars is more direct hence the -1
          return previous_level[(cluster-1,time-1)]
      else:
        return variable[(c-1,time-1)]
    else:
      return variable[(c, t-1)]

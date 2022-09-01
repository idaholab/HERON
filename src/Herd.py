# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  New HERON workflow for setting up and running DISPATCHES cases
  HEron Runs Dispatches (HERD)
"""
import os.path as path
import json
import sys
import copy
import operator
import pandas as pd
from itertools import compress
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import _utils as hutils

# Nuclear flowsheet function imports
from dispatches.case_studies.nuclear_case.nuclear_flowsheet import build_ne_flowsheet
from dispatches.case_studies.nuclear_case.nuclear_flowsheet import fix_dof_and_initialize

# Import function for the construction of the multiperiod model
from dispatches.case_studies.nuclear_case.multiperiod import build_multiperiod_design

from idaes.core.solvers import get_solver

# append path with RAVEN location
path_to_raven = hutils.get_raven_loc()
sys.path.append(path.abspath(path.join(path_to_raven, 'scripts')))
sys.path.append(path.abspath(path.join(path_to_raven, 'plugins')))
sys.path.append(path_to_raven)

from TEAL.src import CashFlows
from TEAL.src import main as RunCashFlow
from HERON.src.Moped import MOPED

dispatches_model_component_meta={
  "Nuclear-Hydrogen IES: H2 Production, Storage, and Combustion": {
    "pem":{ # will require some transfer function
      "Produces": 'hydrogen',
      "Consumes": 'electricity',
      "Cashflows":{
        "Capacity":{
          "Expressions": 'pem_capacity',
        },
        "Dispatch":{
          "Expressions": ['fs.pem.electricity'],
        },
      },
    },
    "h2tank":{
      "Stores": 'hydrogen',
      "Consumes": {},
      "Cashflows":{
        "Capacity":{
          "Expressions": 'tank_capacity',
          "Multiplier":  2.016e-3, # H2 Molar Mass = 2.016e-3 kg/mol
        },
      },
    },
    "h2turbine":{ # TODO: technically also consumes air, will need to revisit
      "Produces": 'electricity',
      "Consumes": 'hydrogen',
      "Cashflows":{
        "Capacity":{
          "Expressions": 'h2_turbine_capacity',
        },
        "Dispatch":{
          "Expressions": ['fs.h2_turbine.turbine.work_mechanical',
            'fs.h2_turbine.compressor.work_mechanical'],
          "Multiplier":  [-1, -1] # extra multiplier to ensure correct sign
        },
      },
    },
    "electricity_market":{
      "Demands":  'electricity',
      "Consumes": {},
      "Cashflows":{
        "Dispatch":{
          "Expressions": ['fs.np_power_split.np_to_grid_port.electricity',
            'fs.h2_turbine.turbine.work_mechanical',
            'fs.h2_turbine.compressor.work_mechanical'],
          "Multiplier":  [1e-3, -1e-6, -1e-6] # NOTE: h2 turbine is in W, convert to kW
        },
      },
    },
    "h2_market":{
      "Demands":  'hydrogen',
      "Consumes": {},
      "Cashflows":{
        "Dispatch":{
          "Expressions": ['fs.h2_tank.outlet_to_pipeline.flow_mol'],
          "Multiplier":  [7.2576] # convert 1/s to 1/hr and 2.016e-3 kg/mol -> kg/hr
        },
      },
    },
  },
}

class HERD(MOPED):
  """
    Main class used for communicating between HERON and a
    DISPATCHES case
  """

  def __init__(self):
    """
      Initializes main class by calling parent class MOPED
    """
    # running the init for MOPED first to initialize empty params
    super().__init__()
    self._dmdl = None # Pyomo model specific to DISPATCHES (different from _m)
    self._dispatches_model_template = None # Template of DISPATCHES Model for HERON comparison
    self._metrics = None # TEAL metrics, summed expressions
    self._results = None # results from Dispatch solve
    self._simTime = 0
    self._num_samples = 0
    self._simTimeSet = range(0)
    self._demand_meta = {}

    # some testing stuff
    self._testMode = False
    self._testYears = []
    self._testProjLife = 20

  def buildEconSettings(self, verbosity=0):
    """
      Builds TEAL economic settings for running cashflows
      @ In, verbosity, int or string, verbosity settings for TEAL
      @ out, None
    """
    # checking for a specific case - testing the DISPATCHES base Nuclear Case
    if self._case.name == 'test_dispatches_wJSON':
      self._testMode = True
      # testing for 20 year project life, override because it doesnt match LMP JSON signal
      globalEcon = getattr(self._case,"_global_econ")
      globalEcon["ProjectTime"] = self._testProjLife
      # intended years to test out (2022-2031, use same data. 2032-2041, use same data)
      self._testYears = [2022, 2032]
    # now run parent method
    super().buildEconSettings(verbosity)
    self._num_samples = getattr(self._case,"_num_samples")

  def buildComponentMeta(self):
    """
      Collect component attributes, actions into one dictionary
      @ In, None
      @ Out, None
    """
    self._m = pyo.ConcreteModel(name=self._case.name)
    # Considering all components in analysis to build a full pyomo solve
    for comp in self._components:
      self._component_meta[comp.name] = {}
      for prod in getattr(comp, "_produces"): # NOTE Cannot handle components that produce multiple things
        self.getComponentActionMeta(comp, prod, "Produces")
      for sto in getattr(comp, "_stores"):
        self.getComponentActionMeta(comp, sto, "Stores")
      for dem in getattr(comp, "_demands"): # NOTE Cannot handle components that demand multiple things
        self.getComponentActionMeta(comp, dem, "Demands")

  def _getDemandData(self):
    """
      Builds cashflow meta used in cashflow component construction
      @ In, None
      @ Out, None
    """
    for comp in self._components:
      if comp.name.split('_')[-1] == 'market':
        # NOTE: only considering one resource in demand
        demand = getattr(comp, "_demands")[0]
        demand_cap  = getattr(demand, "_capacity")
        demand_type = demand_cap.type
        demand_vp   = getattr(demand_cap, "_vp")

        if demand_type == "FixedValue":
          demand_signal  = getattr(demand_vp, "_parametric") * -1

        elif demand_type == 'SyntheticHistory':
          signal = getattr(demand_vp, "_var_name")
          demand_signal = self.loadSyntheticHistory(signal, 1)

        self._demand_meta[comp.name] = {}
        self._demand_meta[comp.name]["Demand"] = demand_signal

  def getComponentActionMeta(self, comp, action, action_type=None):
    """
      Checks the capacity type, dispatch type, and resources involved for each component
      to build component_meta. Repurposed from MOPED, doesn't create Pyomo objects
      @ In, comp, HERON component
      @ In, action, HERON produces/demands/stores node
      @ In, action_type, str, name of HERON component action type
      @ Out, None
    """
    # Organizing important aspects of problem for later access
    resource = getattr(action, "_capacity_var")
    capacity = getattr(action, "_capacity")
    mode     = getattr(capacity, "type")
    value    = getattr(capacity, "_vp")

    consumes = bool(getattr(action, "_consumes")) if hasattr(action, "_consumes") else False

    # Multiplier plays important role in capacity node, especially for VRE's
    capmult = getattr(capacity, '_multiplier')
    capacity_mult = 1 if capmult is None else capmult
    # capacity_mult = -1*capacity_mult if capacity_mult<0 else capacity_mult #NOTE: need this?

    # saving resource under action type, e.g. "Produces": "electricity"
    self._component_meta[comp.name][action_type] = resource
    self._component_meta[comp.name]['Consumes'] = {}
    self._component_meta[comp.name]['Dispatch'] = getattr(action, "_dispatchable")

    # save optimization parameters
    if mode in ('OptBounds', 'FixedValue'):
      self.raiseADebug(f'|Building pyomo capacity {mode} for {comp.name}|')
      self._component_meta[comp.name][mode] = getattr(value, "_parametric") * capacity_mult

    # sample synthetic histories
    elif mode == 'SyntheticHistory':
      self.raiseADebug(f'|Building pyomo parameter with synthetic histories for {comp.name}|')
      synthHist = self.loadSyntheticHistory( getattr(value, "_var_name"), capacity_mult ) # runs external ROM load
      self._component_meta[comp.name][mode] = synthHist

    # cannot do sweep values yet
    elif mode == 'SweepValues': # TODO Add capability to handle sweepvalues, maybe multiple pyo.Params?
      raise IOError('MOPED does not currently support sweep values option')

    # NOTE not all producers consume
    # TODO should we handle transfer functions here?
    if consumes:
      for con in getattr(action, "_consumes"):
        self._component_meta[comp.name]['Consumes'][con] = getattr(action, "_transfer")

  def loadSyntheticHistoryFromJSON(self):
    """
      Load synthetic history data specifically from a JSON file (very specific for testing)
    """
    # paths to LMP signal JSON within DISPATCHES
    proj_path = path.dirname(path_to_raven)
    disp_path = path.join(proj_path, "dispatches/dispatches/case_studies/nuclear_case/")
    lmp_path  = path.abspath( path.join(disp_path, "lmp_signal.json") )

    # loading JSON data
    with open(lmp_path, encoding='utf-8') as fp:
      synthHist = json.load(fp)

    # data is the same for 2022-2031, and 2032-2041
    #   to save on # of variables, just duplicate LMP values
    testYears = self._testYears
    n_days = len(synthHist['0']['2020'].keys())
    n_time = len(synthHist['0']['2020']['1'].keys()) -1

    # building array of simulation years
    projLifeRange = np.arange( testYears[0],   # year[0]-1 is the construction year
                  testYears[0] + self._testProjLife) # full project time
    set_years_map = np.array([testYears[sum(y>=testYears)-1] for y in projLifeRange])
     # this array looks like [2022, 2022, ...., 2032, 2032, ...]

    # creating set of scenarios/realizations/samples
    n_scenarios = self._num_samples
    set_scenarios = list( range(n_scenarios) )

    # we have to rebuild the LMP signal, using set_years as a map for when to duplicate data
    if len(testYears) < self._testProjLife:
      fullHist = {}
      # looping through scenarios, re-building LMP signal if necessary
      for r in set_scenarios:
        # output dict key is actual year, input dict key is mapped year with duplicates
        fullHist[str(r)] = {str(y): synthHist[str(r)][str(i)]
                                for i,y in zip(set_years_map, projLifeRange)}
    else:
      # same dictionary
      fullHist = synthHist

    sets = [set_years_map, testYears, projLifeRange, set_scenarios]
    setLengths = [len(set_years_map), n_days, n_time] # hard-coded for now, just to check

    return fullHist, sets, setLengths

  def loadSyntheticHistory(self, signal, multiplier):
    """
      Loads synthetic history for a specified signal,
      also sets yearly hours and pyomo indexing sets.
      Calls the parent method and restructures dictionary
      to match DISPATCHES format.
      @ In, signal, string, name of signal to sample
      @ Out, synthetic_data, dict, contains data from evaluated ROM
    """
    # calling parent method for loading synthetic history
    testJSON = (signal == 'dispatches-test' and self._testMode)
    if testJSON:
      synthHist, [set_years_map, set_years, projYears, set_scenarios], [n_years, n_days, n_hours] \
                = self.loadSyntheticHistoryFromJSON()
    else: # normal extraction
      synthHist = super().loadSyntheticHistory(signal, multiplier)
      n_years, n_days, n_hours = synthHist['Realization_1'].shape
      set_scenarios = range(sum("Realization" in entry for entry in synthHist.keys()))
      set_years = list(synthHist['years'])
      set_years_map = set_years
      projYears = set_years
    # set data
    set_days = range(1, n_days + 1)  # to appease Pyomo, indexing starts at 1
    set_time = range(1, n_hours + 1) # to appease Pyomo, indexing starts at 1
    # double check years
    # if len(set_years) != n_years:
    #   raise IOError("Discrepancy in number of years within Synthetic History")

    # restructure the synthetic history dictionary to match DISPATCHES
    newHist = {}
    newHist['signals'] = {}
    # converting to dictionary that plays nice with DISPATCHES/IDAES
    if testJSON:
      for scenario in set_scenarios:
        newHist['signals'][scenario] = {year: {day: {hour:
                                        synthHist[str(scenario)][str(year)][str(day)][str(hour)]
                                                      for hour in set_time}
                                              for day in set_days}
                                      for year in projYears}
    else:
      for key, data in synthHist.items():
        # assuming the keys are in format "Realization_i"
        if "Realization" in key:
          # realizations known as scenarios in DISPATCHES, index starting at 0
          k = int( key.split('_')[-1] )
          # years indexed by integer year (2020, etc.)
          # clusters and hours indexed starting at 1
          newHist['signals'][set_scenarios[k-1]] = {year: {cluster: {hour:
                                                                          data[y, cluster-1, hour-1]
                                                                    for hour in set_time}
                                                          for cluster in set_days}
                                                    for y, year in enumerate(set_years)}
    # save set time data for use within DISPATCHES
    newHist["sets"] = {}
    newHist["sets"]["set_scenarios"] = list(set_scenarios) # DISPATCHES wants this as a list
    newHist["sets"]["set_years"] = set_years # DISPATCHES wants this as a list
    newHist["sets"]["set_days"]  = set_days  # DISPATCHES wants this as a range
    newHist["sets"]["set_time"]  = set_time  # DISPATCHES wants this as a range
    newHist["sets"]["set_years_map"] = set_years_map # used only for tests
    newHist["sets"]["projYears"] = projYears # used only for tests
    #TODO: need weights_days - how many days does each cluster represent?
    if testJSON:
      newHist["weights_days"] = {year: {cluster:
                                            synthHist[str(0)][str(year)][str(cluster)]["num_days"]
                                      for cluster in set_days}
                                for year in set_years}
    elif n_days == 2:
      tmp_weights = [183, 182] # just guessing for now, need this info from ARMA model?
      newHist["weights_days"]  = {year: {cluster: tmp_weights[cluster-1]
                                      for cluster in set_days}
                                for year in set_years}
    else:
      raise IOError("HERD not set up for more than 2 clusters yet when using ROM.")

    return newHist

  def _checkDispatchesCompatibility(self):
    """
      Checks HERON components to match compatibility with available
      DISPATCHES flowsheets.
    """
    # TODO: check for financial params/inputs?
    heron_comp_list = list( self._component_meta.keys() ) # current list of HERON components
    self.raiseADebug(f'|Checking compatibility between HERON and available DISPATCHES cases|')

    # check that HERON input file contains all components needed to run DISPATCHES case
    # using naming convention: d___ corresponds to DISPATCHES, h___ corresponds to HERON
    dispatches_model_template = copy.deepcopy(dispatches_model_component_meta)
    for dName, dModel in dispatches_model_template.items():
      dispatches_comp_list    = list( dModel.keys() )
      incompatible_components = [dComp not in heron_comp_list for dComp in dispatches_comp_list]

      # 1. first check: do component names match? NOTE: shouldn't really care, just easier to check
      if sum(incompatible_components) > 0:
        # print list of components missing from HERON input
        missing_comps = list(compress(dispatches_comp_list, incompatible_components))
        message  = f'HERON components do not match DISPATCHES Model: {dName}\n'
        message +=  'Components missing from HERON XML input file: '
        message += ', '.join(missing_comps)
        raise IOError(message)

      # now let's check individual component actions
      for dComp in dispatches_comp_list:
        hCompDict = self._component_meta[dComp]  # HERON component dict, same name as DISPATCHES
        #FIXME: temp fix to not check for Cashflows just yet
        if 'Cashflows' in dModel[dComp].keys():
          del dModel[dComp]['Cashflows']
        dispatches_actions_list = list(dModel[dComp].keys())
        incompatible_actions = [dAction not in hCompDict.keys()
                                      for dAction in dispatches_actions_list]

        # 2. second check: do the components have the necessary actions?
        if sum(incompatible_actions) > 0:
          missing_actions = list(compress(dispatches_actions_list, incompatible_actions))
          message = f'HERON Component {dComp} is missing the follow attributes: '
          message += ', '.join(missing_actions)
          raise IOError(message)

        # 3. third check: do the HERON component resources match the DISPATCHES ones?
        mismatched_actions = []
        for dAction, dResource in dModel[dComp].items():
          hAction = hCompDict[dAction]  # HERON component's action, might be a dict or str
          if isinstance(hAction, dict):
            hResource = list(hAction.keys())[0] if hAction else {}
            mismatched_actions.append(hResource != dResource)
          else:
            mismatched_actions.append(hAction != dResource)

        if sum(mismatched_actions) > 0:
          message = f'Attributes of HERON Component {dComp} do not match DISPATCHES case: '
          message += ', '.join( list(compress(dispatches_actions_list, mismatched_actions)) )
          raise IOError(message)
      break

    self.raiseADebug(f'|HERON Case is compatible with {dName} DISPATCHES Model|')
    self._dispatches_model_template = dispatches_model_component_meta[dName] # NOTE: NOT using copy

  def _addSetsToPyomo(self):
    """
      Create new DISPATCHES Pyomo Model with available data
    """
    market_synthetic_history = self._component_meta['electricity_market']['SyntheticHistory']

    # transferring information on Sets
    sets = market_synthetic_history['sets']
    self._dmdl.set_time  = sets['set_time']
    self._dmdl.set_days  = sets['set_days']
    self._dmdl.set_years = sets['set_years']
    self._dmdl.set_years_map = sets["set_years_map"]
    self._dmdl.set_scenarios = sets['set_scenarios']

    # transferring information on LMP Synthetic History signal
    # TODO: here we use TOTALLOAD data as a stand-in, should actually be price signals
    self._dmdl.LMP = market_synthetic_history['signals']

    # transferring information on weightings
    self._dmdl.weights_days = market_synthetic_history['weights_days']

    # NOTE: equal probability for all scenarios
    self._dmdl.weights_scenarios = {s:1/len(self._dmdl.set_scenarios)
                                        for s in self._dmdl.set_scenarios}

  def _buildDispatchesModel(self):
    """
      Build dispatches model
    """

    self._addSetsToPyomo()

    mdl_flowsheet = build_ne_flowsheet # pointing to the imported DISPATCHES nuclear flowsheet
    mdl_init  = fix_dof_and_initialize # pointing to the imported DISPATCHES fix/init method
    mdl_unfix = self.unfixDof # we add a method to unfix certain DoFs based on DISPATCHES jupyter notebooks

    # NOTE: within the build process, a tmp JSON file is created in wdir...
    build_multiperiod_design(self._dmdl,
                         flowsheet=mdl_flowsheet,
                         initialization=mdl_init,
                         unfix_dof=mdl_unfix,
                         multiple_days=True,
                         multiyear=True,
                         stochastic=True,
                         verbose=False)

    # list of initialized TEAL components; filters out HERON components that don't have cash flows
    teal_components, heron_components = self._initializeCashFlows()

    for s in self._dmdl.set_scenarios:
      # Build the connecting constraints
      self.buildConnectingConstraints(self._dmdl.scenario[s],
                                  set_time=self._dmdl.set_time,
                                  set_days=self._dmdl.set_days,
                                  set_years=self._dmdl.set_years)

      # Append cash flow expressions
      for hComp, tComp in zip(heron_components, teal_components): #this zip might be danger
        if hComp.name not in self._dispatches_model_template.keys():
          continue # skip components within HERON that are NOT defined in DISPATCHES template

        self._createCashflowsForDispatches(self._dmdl.scenario[s], hComp, tComp, s)

      # Hydrogen demand constraint.
      # Divide the RHS by the molecular mass to convert kg/s to mol/s
      scenario = self._dmdl.scenario[s]
      @scenario.Constraint(self._dmdl.set_time, self._dmdl.set_days, self._dmdl.set_years)
      def hydrogen_demand_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.h2_tank.outlet_to_pipeline.flow_mol[0] \
                  <= self._demand_meta['h2_market']["Demand"] / 2.016e-3 # convert from kg to mol

    self._cf_components.extend(teal_components)
    self._metrics = RunCashFlow.run(self._econ_settings, self._cf_components, {}, pyomoVar=True)

    with open("HERD_output.txt", "w") as text_file:
      # text_file.write(str(m.npv.expr))
      text_file.write(str(self._metrics['NPV']))

  def _initializeCashFlows(self):

    teal_components  = []
    heron_components = []
    for hComp in self._components:
      if hComp._economics._cash_flows == [] or hComp.name not in self._dispatches_model_template.keys():
        continue
      # blank TEAL cash flow
      tealComp   = CashFlows.Component()
      tealParams = {"name": hComp.name} # beginning of params dict for TEAL

      self.raiseADebug(f'Setting component lifespan for {hComp.name}')
      tealParams['Life_time'] = self._cf_meta[hComp.name]['Lifetime']
      tealComp.setParams(tealParams)
      teal_components.append(tealComp)

      heron_components.append(hComp)
    return teal_components, heron_components

  def _createCashflowsForDispatches(self, scenario, hComp, tComp, scenario_ind):
    """
      Identify the correct DISPATCHES expressions to use as TEAL drivers
    """
    # collection of cash flows for given HERON component
    CF_collection = []
    CF_meta = self._cf_meta[hComp.name]

    # loop through all cashflows for given component
    for cf, value in CF_meta.items():
      if cf == 'Lifetime':
        continue

      if cf == 'Capex':
        # Capex is the most complex to handle generally due to amort
        self.raiseADebug(f'Generating Capex cashflow for {hComp.name}')
        capex_params = CF_meta['Capex Params']
        capex_driver = CF_meta['Capex Driver']

        # getting capacity Pyomo object if capex_driver is not defined
        if capex_driver is None:
          capex_driver, mult = self._getCapacityFromDispatchesModel(scenario,
                                              self._dispatches_model_template[hComp.name])
          # mult defaults to 1
          value *= mult
        # generate TEAL capex cashflow
        capex = self.createCapex(tComp,
                              alpha=value,
                              capacity=capex_driver,
                              unique_params=capex_params)
        CF_collection.append(capex)
        # define depreciation
        depreciation = CF_meta['Deprec']
        if depreciation is not None:
          capex.setAmortization('MACRS', depreciation)
          amorts = getattr(tComp, "_createDepreciation")(capex)
          CF_collection.extend(amorts)
        print(f"----{hComp.name}---- Capex Driver : {capex_driver}")

      elif cf == "Yearly":
        self.raiseADebug(f'Generating Yearly OM cashflow for {hComp.name}')
        yearly_params = CF_meta['Yearly Params']
        yearly_driver = CF_meta['Yearly Driver']
        if yearly_driver is None:
          # assuming that yearly fixed OM is based on capacity
          yearly_driver, mult = self._getCapacityFromDispatchesModel(scenario,
                                              self._dispatches_model_template[hComp.name])

        yearly = self.createRecurringYearly(tComp, value, yearly_driver, yearly_params)
        CF_collection.append(yearly)
        print(f"----{hComp.name}---- Yearly Driver : {yearly_driver}")

      elif cf == "Dispatching":
        # Here value can be a np.array as well for ARMA grid pricing
        self.raiseADebug(f'Generating dispatch OM cashflow for {hComp.name}')
        dispatching_params = CF_meta['Dispatching Params']
        dispatch_driver    = CF_meta['Dispatch Driver']
        if dispatch_driver is None:
          # these should return nested lists
          dispatch_driver = self._getDispatchFromDispatchesModel(scenario, hComp,
                                        self._dispatches_model_template[hComp.name])
        # check for alpha as a time series
        if isinstance(value, dict):
          value = self.reshapeAlpha(value)
          value = value[scenario_ind, :, :]

        hourly = self.createRecurringHourly(tComp, value,
                                        dispatch_driver, scenario_ind,
                                        dispatching_params)
        CF_collection.append(hourly)
        print(f"----{hComp.name}---- Dispatch Driver : {dispatch_driver[2,8]}")

    tComp.addCashflows(CF_collection)

  def _getCapacityFromDispatchesModel(self, mdl, dComp):
    """
    """
    cashflows_dict = dComp['Cashflows']
    capacity_dict  = cashflows_dict['Capacity']
    capacity_str   = capacity_dict['Expressions']

    capacity_driver = operator.attrgetter(capacity_str)(mdl)
    mult = capacity_dict['Multiplier'] if 'Multiplier' in capacity_dict.keys() else 1

    return capacity_driver, mult

  def _getDispatchFromDispatchesModel(self, mdl, hComp, dComp):
    """
    """
    # project life time + 1, first year has to be 0 for recurring cash flows
    projectLife = int( operator.attrgetter("_case._global_econ")(self)['ProjectTime'] )
    projectLife += 1

    # time indeces for HERON/TEAL
    n_hours = len(self._dmdl.set_time)
    n_days  = len(self._dmdl.set_days)
    n_years = len(self._dmdl.set_years)
    n_hours_per_year = n_hours * n_days # sometimes number of days refers to clusters < 365

    set_years_map = np.hstack([0, self._dmdl.set_years_map])

    # # FIXME: this check should happen upstream somewhere? does HERON check?
    # if n_years < projectLife - 1:
    #   raise IOError("Project Life exceeds number of years in available dispatch")

    # template array for holding dispatch Pyomo expressions/objects
    dispatch_array = np.zeros((projectLife, n_hours_per_year), dtype=object)
    dispatch_type = self._component_meta[hComp.name]['Dispatch']

    # TODO: there should be a more robust check of dispatch type, further upstream
    if dispatch_type == "fixed":
      self.raiseADebug("Dispatch type is fixed, this should be set through fix/unfix methods")
      return None, None

    # extract Pyomo expressions from DISPATCHES objects
    cashflows_dict = dComp['Cashflows']
    dispatch_dict  = cashflows_dict['Dispatch']
    dispatch_strs  = dispatch_dict['Expressions']

    # time indeces for DISPATCHES, as array of tuples
    indeces = np.array([tuple(i) for i in mdl.period_index], dtype="i,i,i")
    time_shape = (n_years, n_hours_per_year) # reshaping the tuples array to match HERON dispatch
    indeces = indeces.reshape(time_shape)

    # extra multipliers specific to DISPATCHES (e.g., have to multiply turbine work done by -1)
    dMults = dispatch_dict['Multiplier'] \
                if 'Multiplier' in dispatch_dict.keys() \
                else np.ones(len(dispatch_strs)) # defaults to just 1
    pcount = -1
    for p,pyear in enumerate(set_years_map):
      if pyear == 0:
        continue

      if pyear > set_years_map[p-1]:
        pcount +=1

      for time in range(n_hours_per_year):
        ind = tuple(indeces[pcount,time])
        # looping through all DISPATCHES variables pertaining to this specific dispatch
        #   e.g., turbine costs due to work done by turbine + compressor, separate variables
        dispatch_driver = 0
        for ds, dStr in enumerate(dispatch_strs):
          dispatch_driver += operator.attrgetter(dStr)(mdl.period[ind])[0] * dMults[ds]

        # getting weights for each day/cluster
        dy, yr = ind[1:]
        weight = self._dmdl.weights_days[yr][dy]  # extracting weight for year + day

        # storing individual Pyomo dispatch
        dispatch_array[p, time] = dispatch_driver * weight

    return dispatch_array

  def reshapeAlpha(self, alpha):
    """
      This is different because DISPATCHES wants LMP signal in a different format.
    """
    projectLife = int( operator.attrgetter("_case._global_econ")(self)['ProjectTime'] )
    projectLife += 1

    signal = alpha['signals']
    set_scenarios = alpha['sets']['set_scenarios']
    set_years = alpha['sets']['set_years']
    set_days  = alpha['sets']['set_days']
    set_time  = alpha['sets']['set_time']
    projYears = alpha['sets']['projYears']

    # time indeces for HERON/TEAL
    n_hours = len(alpha['sets']['set_time'])
    n_days  = len(alpha['sets']['set_days'])
    n_scenarios = len(set_scenarios)
    n_hours_per_year = n_hours * n_days # sometimes number of days refers to clusters < 365

    # plus 1 to year term to allow for 0 recurring costs during build year
    reshaped_alpha = np.zeros([n_scenarios, projectLife, n_hours_per_year])

    for real in set_scenarios:
      # it necessary to have alpha be [real,year,hour] instead of [real,year, cluster, hour]
      realized_alpha = [[signal[real][y][d][t] \
                                    for d in set_days
                                      for t in set_time]
                                        for y in projYears] #shape here is [year, hour]
      # first column of 2nd axis is 0 for project year 0
      reshaped_alpha[real,1:,:] = realized_alpha

    return reshaped_alpha

  def unfixDof(self, ps, **kwargs):
    """
    This function unfixes a few degrees of freedom for optimization.
    This particular method is taken from the DISPATCHES jupyter notebook
    found in "dispatches/dispatches/models/nuclear_case/flowsheets"
    titled "multiperiod_design_pricetaker"
      @In: ps: Pyomo model for period within a given scenario
    """
    # Set defaults in case options are not passed to the function
    options = kwargs.get("options", {})
    air_h2_ratio = options.get("air_h2_ratio", 10.76)

    # Unfix the electricity split in the electrical splitter
    ps.fs.np_power_split.split_fraction["np_to_grid", 0].unfix()

    # Unfix the holdup_previous and outflow variables
    ps.fs.h2_tank.tank_holdup_previous.unfix()
    ps.fs.h2_tank.outlet_to_turbine.flow_mol.unfix()
    ps.fs.h2_tank.outlet_to_pipeline.flow_mol.unfix()

    # Unfix the flowrate of air to the mixer
    ps.fs.mixer.air_feed.flow_mol.unfix()

    # Add a constraint to maintain the air to hydrogen flow ratio
    ps.fs.mixer.air_h2_ratio = pyo.Constraint(
                expr=ps.fs.mixer.air_feed.flow_mol[int(0)] ==
                      air_h2_ratio * ps.fs.mixer.hydrogen_feed.flow_mol[int(0)])

    # Set bounds on variables. A small non-zero value is set as the lower
    # bound on molar flowrates to avoid convergence issues
    ps.fs.pem.outlet.flow_mol[0].setlb(0.001)

    ps.fs.h2_tank.inlet.flow_mol[0].setlb(0.001)
    ps.fs.h2_tank.outlet_to_turbine.flow_mol[0].setlb(0.001)
    ps.fs.h2_tank.outlet_to_pipeline.flow_mol[0].setlb(0.001)

    ps.fs.translator.inlet.flow_mol[0].setlb(0.001)
    ps.fs.translator.outlet.flow_mol[0].setlb(0.001)

    ps.fs.mixer.hydrogen_feed.flow_mol[0].setlb(0.001)

  def buildConnectingConstraints(self, scenario, set_time, set_days, set_years):
    """
    This function declares the first-stage variables or design decisions,
    adds constraints that ensure that the operational variables never exceed their
    design values, and adds constraints connecting variables at t - 1 and t
    """
    # Declare first-stage variables (Design decisions)
    scenario.pem_capacity = pyo.Var(within=pyo.NonNegativeReals,
                          doc="Maximum capacity of the PEM electrolyzer (in kW)")
    scenario.tank_capacity = pyo.Var(within=pyo.NonNegativeReals,
                          doc="Maximum holdup of the tank (in mol)")
    scenario.h2_turbine_capacity = pyo.Var(within=pyo.NonNegativeReals,
                          doc="Maximum power output from the turbine (in W)")

    # Ensure that the electricity to the PEM elctrolyzer does not exceed the PEM capacity
    @scenario.Constraint(set_time, set_days, set_years)
    def pem_capacity_constraint(blk, t, d, y):
      return blk.period[t, d, y].fs.pem.electricity[int(0)] <= scenario.pem_capacity

    # Ensure that the final tank holdup does not exceed the tank capacity
    @scenario.Constraint(set_time, set_days, set_years)
    def tank_capacity_constraint(blk, t, d, y):
      return blk.period[t, d, y].fs.h2_tank.tank_holdup[int(0)] <= scenario.tank_capacity

    # Ensure that the power generated by the turbine does not exceed the turbine capacity
    @scenario.Constraint(set_time, set_days, set_years)
    def turbine_capacity_constraint(blk, t, d, y):
      return (
          - blk.period[t, d, y].fs.h2_turbine.turbine.work_mechanical[int(0)]
          - blk.period[t, d, y].fs.h2_turbine.compressor.work_mechanical[int(0)] <=
          scenario.h2_turbine_capacity  )

    # Connect the initial tank holdup at time t with the final tank holdup at time t - 1
    @scenario.Constraint(set_time, set_days, set_years)
    def tank_holdup_constraints(blk, t, d, y):
      if t == 1:
        # Each day begins with an empty tank
        return (
          blk.period[t, d, y].fs.h2_tank.tank_holdup_previous[int(0)] == 0  )
      else:
        # Initial holdup at time t = final holdup at time t - 1
        return (
          blk.period[t, d, y].fs.h2_tank.tank_holdup_previous[int(0)] ==
          blk.period[t - 1, d, y].fs.h2_tank.tank_holdup[int(0)]   )

  def _addNonAnticipativityConstraints(self):
    """
      Add non anticipativity constraints
    """
    # temporary object pointing to model
    dmdl = self._dmdl

    # Add non-anticipativity constraints
    dmdl.pem_capacity = pyo.Var(within=pyo.NonNegativeReals,
                        doc="Design PEM capacity (in kW)")
    dmdl.tank_capacity = pyo.Var(within=pyo.NonNegativeReals,
                          doc="Design tank capacity (in mol)")
    dmdl.h2_turbine_capacity = pyo.Var(within=pyo.NonNegativeReals,
                                doc="Design turbine capacity (in W)")

    @dmdl.Constraint(dmdl.set_scenarios)
    def non_anticipativity_pem(blk, s):
      return blk.pem_capacity == blk.scenario[s].pem_capacity

    @dmdl.Constraint(dmdl.set_scenarios)
    def non_anticipativity_tank(blk, s):
      return blk.tank_capacity == blk.scenario[s].tank_capacity

    @dmdl.Constraint(dmdl.set_scenarios)
    def non_anticipativity_turbine(blk, s):
      return blk.h2_turbine_capacity == blk.scenario[s].h2_turbine_capacity

  def _addObjective(self):
    """
      Adding objective function to model
    """
    self._dmdl.obj = pyo.Objective(expr=self._metrics['NPV'], sense=pyo.maximize)

  def _solveDispatchesModel(self):
    """
      Solving model
    """
    # Define the solver object. Using the default solver: IPOPT
    solver = get_solver()

    # Solve the optimization problem
    self._results = solver.solve(self._dmdl, tee=True)

  def _exportResults(self):

    mdl = self._dmdl

    print(f'Optimal NPV is ------------------- $B {pyo.value(mdl.obj) * 1e-9} ')
    print(f'--- Optimal PEM capacity is        {pyo.value(mdl.pem_capacity)  * 1e-3} ??')
    print(f'--- Optimal H2 Tank capacity is    {pyo.value(mdl.tank_capacity) * 2.016e-3} ??')
    print(f'--- Optimal H2 Turbine capacity is {pyo.value(mdl.h2_turbine_capacity) * 1e-6} ??')

  def run(self):
    """
      Runs the workflow
      @ In, None
      @ Out, None
    """
    # original workflow from MOPED
    self.buildEconSettings()  # overloaded method
    self.buildComponentMeta() # overloaded method
    self.buildCashflowMeta()  # MOPED method
    self.collectResources()   # MOPED method (TODO: needed?)

    # new workflow for DISPATCHES
    self._checkDispatchesCompatibility()
    self._getDemandData()

    # building the Pyomo model using DISPATCHES
    self._dmdl = pyo.ConcreteModel(name=self._case.name)
    self._buildDispatchesModel()
    self._addNonAnticipativityConstraints()
    self._addObjective()
    self._solveDispatchesModel()
    self._exportResults()

    # now we need to import DISPATCHES
    #  X. load in the synthetic histories to DISPATCHES
    #  X. create weights
    #  X. create overall Pyomo model
    #  X. set global parameters from HERON
    #  X. build DISPATCHES multiperiod
    #      X. point to existing flowsheet
    #      X. point to existing init/fix method
    #      X. **NEW** overload unfix method??
    #  X. run a loop over scenarios
    #      X. build constraints (**NEW** import some params from HERON??) per scenario
    #      X. run append_costs
    #           X.  gather all required Pyomo expressions from DISPATCHES
    #           X. gather all required alphas, multipliers from HERON Input
    #           X. combine both into TEAL cashflows by using Pyomo expressions as the drivers
    #                X. loop through DISPATCHES metadata?
    #  X. add non-anticipativity constraints
    #  X. add objective (sum up npvs)
    #  X. run solver from idaes
    #  10. output to CSV?

    # check that NPV summed expression has all correct values



# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  New HERON workflow for setting up and running DISPATCHES cases
  (HE)RON (R)uns (D)ISPATCHES (HERD)
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
path_to_raven = hutils.get_raven_loc()
sys.path.append(path.abspath(path.join(path_to_raven, 'scripts')))
sys.path.append(path.abspath(path.join(path_to_raven, 'plugins')))
sys.path.append(path_to_raven)
from ravenframework.utils import xmlUtils
import externalROMloader as ROMloader

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
  "Nuclear Case": {
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
    (HE)RON (R)uns (D)ISPATCHES
    Main class used for communicating between HERON and a
    DISPATCHES case (currently, only nuclear case)
  """

  def __init__(self):
    """
      Initializing HERD class by calling parent class init.
      @ In, None
      @ Out, None
    """
    # running the init for MOPED first to initialize empty params
    super().__init__()

    # extra parameters for HERD
    self._dmdl = None # Pyomo model specific to DISPATCHES (different from _m)
    self._dispatches_model_template = None # Template of DISPATCHES Model for HERON comparison
    self._timeIndexMap = ['years', 'days', 'hours'] # index map to save time sets to dict later
    self._metrics = []     # TEAL metrics, summed expressions
    self._results = None   # results from Dispatch solve
    self._num_samples = 0  # number of samples/scenarios/realizations for easier retrievability
    self._demand_meta = {} # saving demand data to separate dict (in case it is also sampled)
    self._synthhistories = {}  # nested dict of all synthetic histories

    # Testing - using LMP signals from JSON script as used in example Jupyter notebook
    #    intended years to test out (2022-2031, use same data. 2032-2041, use same data)
    self._testMode = False
    self._testSynthYears = []  # the actual years to test out, [2022, 2032]
    self._testProjLife = 0     # intended length of project, regardless of synthyears
    self._testProjectYearRange = [] # actual years of project [2022, 2023, 2024, ...]
    self._testMap_Synth2Proj = [] # map from synth to actual years [2022, 2022, ..., 2032, ...]

  def _setTestTimeSets(self):
    """
      Sets object attributes for time sets specifically for JSON test.
      @ In, None
      @ Out, None
    """
    self._testSynthYears = [2022, 2032]
    self._testProjLife = 20
    # range of years through intended project life (_testSynthYears contained within this set)
    #   year[0]-1 is the construction year
    self._testProjectYearRange =  np.arange(self._testSynthYears[0],
                                            self._testSynthYears[0] + self._testProjLife)
    # array map, same length as project year range but with entries in _testSynthYears
    self._testMap_Synth2Proj = np.array([self._testSynthYears[sum(y>=self._testSynthYears) - 1]
                                            for y in self._testProjectYearRange])

  def buildEconSettings(self, verbosity=0):
    """
      Builds TEAL economic settings for running cashflows
      @ In, verbosity, int or string, verbosity settings for TEAL
      @ out, None
    """
    # checking for a specific case - testing the DISPATCHES base Nuclear Case
    if np.any([source.name == 'dispatches-test' for source in self._sources]):
      self._testMode = True
      self._setTestTimeSets()
      # testing for 20 year project life, override because it doesnt match LMP JSON signal
      globalEcon = getattr(self._case,"_global_econ")
      globalEcon["ProjectTime"] = self._testProjLife # some funky pointer stuff here

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
      Builds cashflow meta specifically for market demand data
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
    elif mode == 'SweepValues': # TODO Add capability to handle sweepvalues
      raise IOError('MOPED does not currently support sweep values option')

    # NOTE not all producers consume
    # TODO should we handle transfer functions here?
    if consumes:
      for con in getattr(action, "_consumes"):
        self._component_meta[comp.name]['Consumes'][con] = getattr(action, "_transfer")

  def sampleFromROM(self, signal, multiplier):
    """
      Loads synthetic history for a specified signal, from given ROM.
      @ In, signal, string, name of signal to sample
      @ In, multiplier, int/float, value to multiply synthetic history evaluations by
      @ Out, synthetic_data, dict, contains data from evaluated ROM
    """
    # NOTE self._sources[0]._var_names are the user assigned signal names in DataGenerators
    source = getattr(self, "_sources")[0]
    source_var_names = getattr(source, "_var_names")

    # check that signal name is available within data generator
    if signal not in source_var_names:
      raise IOError('The requested signal name is not available'
                    'from the synthetic history, check DataGenerators node in input')

    # Initializing ravenROMexternal object gives PATH access to xmlUtils
    target_file = getattr(source, "_target_file")
    runner = ROMloader.ravenROMexternal( binaryFileName=target_file,
                                         whereFrameworkIs=hutils.get_raven_loc())

    # TODO expand to change other pickledROM settings withing this method
    inp = {'scaling': [1]}
    nodes = []
    node = xmlUtils.newNode('ROM', attrib={'name': 'SyntheticHistory', 'subType': 'pickledRom'})
    node.append(xmlUtils.newNode('clusterEvalMode', text=self._eval_mode))
    nodes.append(node)
    runner.setAdditionalParams(nodes)

    # sample realizations from ROM
    synthetic_data = {}
    for real in range(self._num_samples):
      self.raiseAMessage(f'Loading synthetic history for signal: {signal}')
      name = f'Realization_{real + 1}'
      current_realization = runner.evaluate(inp)[0]
      # applying mult to each realization is easier than iteration through dict object later
      current_realization[signal] *= multiplier
      if self._eval_mode == 'full':
        # reshape so that a filler cluster index is made
        current_realization[signal] = np.expand_dims(current_realization[signal], axis=1)
      synthetic_data[name] = current_realization[signal]

    # saving index map, often looks like ["Year", "ROM Cluster", "Hour"]
    synthetic_data['indexMap'] = current_realization['_indexMap'][0][signal]

    # get time set data from ROM realizations
    years, days, hours = self._timeIndexMap # defined in __init__, order matters
    for ind in synthetic_data['indexMap']:
      # for year set, we truncate based on desired Project Time (28 yrs available)
      if ind.lower() in 'years':
        projLife = int( getattr(self._case, '_global_econ')['ProjectTime'] ) #TODO: check life < years
        synthetic_data[years] = np.array(current_realization[ind][0:projLife], dtype=int)
      elif ind.lower() in '_rom_cluster_days':
        synthetic_data[days]  = np.array(current_realization[ind] + 1, dtype=int)
      elif ind.lower() in 'hours':
        synthetic_data[hours] = np.array(current_realization[ind] + 1, dtype=int)

    # extracting cluster info from ROM - how many days of year per cluster?
    #    location: runner.rom._segmentROM._macroSteps[2018]._clusterInfo['map']
    #    using attrgetter to get rid of "access to protected member" warning
    cluster_steps = operator.attrgetter('_segmentROM._macroSteps')(runner.rom)
    synthetic_data['weights_days'] = {}
    for year in synthetic_data[years]:
      synthetic_data['weights_days'][year] = {}
      for cluster in synthetic_data[days]:
        cluster_map = operator.attrgetter('_clusterInfo')(cluster_steps[year])['map']
        index = int(cluster-1)
        synthetic_data['weights_days'][year][cluster] = len(cluster_map[index])

    # check to see if clusters add up to 365
    cluster_sums = [sum(cluster
                      for _, cluster in synthetic_data['weights_days'][y].items())
                        for y in synthetic_data[years]]

    if sum(cluster_sums) != 365 * len(synthetic_data[years]):
      raise IOError('ROM cluster weights do not add to 365 for all provided years.')

    # check that evaluation mode is either clustered or full
    if self._eval_mode not in ['clustered', 'full']:
      raise IOError('Improper ROM evaluation mode detected, try "clustered" or "full".')
    return synthetic_data

  def loadSyntheticHistoryFromJSON(self):
    """
      Load synthetic history data specifically from a JSON file (very specific for testing)
      @ In, None
      @ Out, None
    """
    # paths to LMP signal JSON within DISPATCHES
    # TODO: move a copy of this file to HERD?
    proj_path = path.dirname(path_to_raven)
    disp_path = path.join(proj_path, "dispatches/dispatches/case_studies/nuclear_case/")
    lmp_path  = path.abspath( path.join(disp_path, "lmp_signal.json") )

    # loading JSON data
    with open(lmp_path, encoding='utf-8') as fp:
      synthHist = json.load(fp)

    # data is the same for 2022-2031, and 2032-2041
    #   to save on # of variables, just duplicate LMP values
    synthYears = self._testSynthYears # actual years to gather data from e.g., [2022, 2032]

    # building array of simulation years
    projLifeRange = self._testProjectYearRange # actual year range for full project [2022->2041]
    set_years_map = self._testMap_Synth2Proj # array looks like [2022, 2022, ...., 2032, 2032, ...]

    # creating set of scenarios/realizations/samples
    n_scenarios = self._num_samples
    set_scenarios = list( range(n_scenarios) )

    # we have to rebuild the LMP signal, using set_years as a map for when to duplicate data
    if len(synthYears) < self._testProjLife:
      fullHist = {}
      # looping through scenarios, re-building LMP signal if necessary
      for r in set_scenarios:
        # output dict key is actual year, input dict key is mapped year with duplicates
        fullHist[str(r)] = {str(y): synthHist[str(r)][str(i)]
                                for i,y in zip(set_years_map, projLifeRange)}
    else:
      # same dictionary
      fullHist = synthHist

    n_days = len(synthHist['0']['2020'].keys())
    n_time = len(synthHist['0']['2020']['1'].keys()) - 1
    fullHist['years'] = synthYears
    fullHist['days']  = range(1, int(n_days + 1) )
    fullHist['hours'] = range(1, int(n_time + 1) )

    return fullHist

  def loadSyntheticHistory(self, signal, multiplier):
    """
      Loads synthetic history for a specified signal, also sets yearly hours.
      Calls the parent method and restructures dictionary to match DISPATCHES format.
      @ In, signal, string, name of signal to sample
      @ Out, synthetic_data, dict, contains data from evaluated ROM
    """
    if signal == 'price' and multiplier == -1:
      multiplier *= -1 # undoing negative multiplier from one step above, price != demand

    # calling parent method for loading synthetic history
    testJSON = (signal == 'dispatches-test' and self._testMode)

    if testJSON:
      synthHist = self.loadSyntheticHistoryFromJSON()
    else: # normal extraction
      synthHist = self.sampleFromROM(signal, multiplier)

    synth_years, synth_days, synth_hours = [synthHist[ind] for ind in self._timeIndexMap]
    synth_scenarios = range(self._num_samples)
    projYearsRange = self._testProjectYearRange
    map_synth2proj = self._testMap_Synth2Proj
    # restructure the synthetic history dictionary to match DISPATCHES
    newHist = {}
    newHist['signals'] = {}
    # converting to dictionary that plays nice with DISPATCHES/IDAES
    if testJSON:
      for scenario in synth_scenarios:
        newHist['signals'][scenario] = {year: {day: {hour:
                                        synthHist[str(scenario)][str(year)][str(day)][str(hour)]
                                                      for hour in synth_hours}
                                              for day in synth_days}
                                      for year in projYearsRange}
    else:
      for key, data in synthHist.items():
        # assuming the keys are in format "Realization_i"
        if "Realization" in key:
          # realizations known as scenarios in DISPATCHES, index starting at 0
          k = int( key.split('_')[-1] )
          # years indexed by integer year (2020, etc.)"sets"'sets'
          # clusters and hours indexed starting at 1
          newHist['signals'][synth_scenarios[k-1]] = {year: {day: {hour: data[y, day-1, hour-1]
                                                                    for hour in synth_hours}
                                                          for day in synth_days}
                                                    for y, year in enumerate(synth_years)}
    # save set time data for use within DISPATCHES
    newHist["sets"] = {}
    newHist["sets"]["synth_scenarios"] = list(synth_scenarios) # DISPATCHES wants this as a list
    newHist["sets"]["synth_years"]  = synth_years # DISPATCHES wants this as a list
    newHist["sets"]["synth_days"]   = synth_days  # DISPATCHES wants this as a range
    newHist["sets"]["synth_hours"]  = synth_hours # DISPATCHES wants this as a range
    newHist["sets"]["map_synth2proj"] = map_synth2proj # used only for tests
    newHist["sets"]["projYearsRange"] = projYearsRange # used only for tests
    # getting weights_days - how many days does each cluster represent?
    if testJSON:
      newHist["weights_days"] = {yr: {cl: synthHist[str(0)][str(yr)][str(cl)]["num_days"]
                                      for cl in synth_days}
                                for yr in synth_years}
    else:
      newHist["weights_days"] = synthHist['weights_days']

    self._synthhistories[signal] = copy.deepcopy(newHist)
    return newHist

  def _checkDispatchesCompatibility(self):
    """
      Checks HERON components to match compatibility with available
      DISPATCHES flowsheets.
      @ In, None
      @ Out, None
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
        #TODO: temp fix to not check for Cashflows just yet
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
      @ In, None
      @ Out, None
    """
    signals = list( self._synthhistories.keys() )

    if 'dispatches-test' in signals:
      market_synthetic_history = self._synthhistories['dispatches-test']
    elif 'price' in signals:
      market_synthetic_history = self._synthhistories['price']
    else:
      raise IOError('Signal name not found in generated synthetic history dictionary')

    # transferring information on Sets
    sets = market_synthetic_history['sets']
    self._dmdl.set_time  = sets['synth_hours']
    self._dmdl.set_days  = sets['synth_days']
    self._dmdl.set_years = sets['synth_years']
    self._dmdl.set_years_map = sets['map_synth2proj'] if self._testMode else sets['synth_years']
    self._dmdl.set_scenarios = sets['synth_scenarios']

    # transferring information on LMP Synthetic History signal
    # TODO: here we use TOTALLOAD data as a stand-in, should actually be price signals
    self._dmdl.LMP = market_synthetic_history['signals']

    # transferring information on weightings
    self._dmdl.weights_days = market_synthetic_history['weights_days']

    # NOTE: equal probability for all scenarios
    self._dmdl.weights_scenarios = {s:1/self._num_samples for s in range(self._num_samples)}

  def _addAdditionalConstraints(self, mdl):
    """
      Method to add additional constraints not included in DISPATCHES flowsheet
      @ In, mdl, Pyomo model
      @ Out, None
    """
    @mdl.Constraint(self._dmdl.set_time, self._dmdl.set_days, self._dmdl.set_years)
    def hydrogen_demand_constraint(blk, t, d, y):
      return blk.period[t, d, y].fs.h2_tank.outlet_to_pipeline.flow_mol[0] \
                <= self._demand_meta['h2_market']["Demand"] / 2.016e-3 # convert from kg to mol

  def _buildDispatchesModel(self):
    """
      Builds full DISPATCHES Pyomo model
      @ In, None
      @ Out, None
    """
    # add time sets to Pyomo model from given synthetic history and desired project lifetime
    self._addSetsToPyomo()

    # pointing to necessary IDAES/DISPATCHES Physics models
    mdl_flowsheet = build_ne_flowsheet # pointing to the imported DISPATCHES nuclear flowsheet
    mdl_init  = fix_dof_and_initialize # pointing to the imported DISPATCHES fix/init method
    mdl_unfix = self.unfixDof # we add a method to unfix certain DoFs based on DISPATCHES jupyter notebooks

    # NOTE: within the build process, a tmp JSON file is created in wdir...
    build_multiperiod_design(self._dmdl,
                         flowsheet=mdl_flowsheet,
                         initialization=mdl_init,
                         unfix_dof=mdl_unfix,
                         unfix_dof_options={},
                         multiple_days=True,
                         multiyear=True,
                         stochastic=True,
                         verbose=False)

    # list of initialized TEAL components; filters out HERON components that don't have cash flows
    teal_components, heron_components = self._initializeCashFlows()

    # looping through all sampled scenarios
    for s in self._dmdl.set_scenarios:
      # Build the connecting constraints
      self.buildConnectingConstraints(self._dmdl.scenario[s],
                                  set_time=self._dmdl.set_time,
                                  set_days=self._dmdl.set_days,
                                  set_years=self._dmdl.set_years)

      # Hydrogen demand constraint (Divide the RHS by the molecular mass to convert kg/s to mol/s)
      self._addAdditionalConstraints(self._dmdl.scenario[s])

      # Append cash flow expressions
      for hComp, tComp in zip(heron_components, teal_components): #this zip might be danger
        # skip components within HERON that are NOT defined in DISPATCHES template
        if hComp.name not in self._dispatches_model_template.keys():
          continue
        # create cashflows using TEAL (capex, yearly or hourly)
        self._createCashflowsForDispatches(self._dmdl.scenario[s], hComp, tComp, s)

      # compute desired metric using TEAL and storing it
      scenario_metric = RunCashFlow.run(self._econ_settings, teal_components, {}, pyomoVar=True)
      self._metrics.append(scenario_metric)
      del scenario_metric

    with open("HERD_output.txt", "w") as text_file:
      text_file.write(str(self._metrics[0]['NPV']))

  def _initializeCashFlows(self):
    """
      Initialize and populate TEAL cash flows for all components
      @ In, None
      @ Out, None
    """
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
      Create and populate TEAL cash flows for all components
      @ In, scenario, Pyomo model for given scenario
      @ In, hComp, HERON component dictionary
      @ In, tComp, TEAL cashflow component object
      @ In, scenario_ind, integer for scenario index
      @ Out, None
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
      Get capacity driver string from DISPATCHES dictionary and use it to extract
      correct capacity Pyomo expression from scenario model.
      @ In, mdl, Pyomo model for given scenario
      @ In, dComp, DISPATCHES component dictionary
      @ Out, None
    """
    cashflows_dict = dComp['Cashflows']
    capacity_dict  = cashflows_dict['Capacity']
    capacity_str   = capacity_dict['Expressions']

    capacity_driver = operator.attrgetter(capacity_str)(mdl)
    mult = capacity_dict['Multiplier'] if 'Multiplier' in capacity_dict.keys() else 1

    return capacity_driver, mult

  def _getDispatchFromDispatchesModel(self, mdl, hComp, dComp):
    """
      Get dispatch driver string from DISPATCHES dictionary and use it to extract
      correct dispatch Pyomo expression from scenario model.
      @ In, mdl, Pyomo model for given scenario
      @ In, hComp, HERON component dictionary
      @ In, dComp, DISPATCHES component dictionary
      @ Out, None
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
      Reshape alpha time series (multiplier for a cash flow) to match DISPATCHES structure.
      @ In, alpha, synthetic history dictionary for time series cash flow
      @ Out, None
    """
    projectLife = int( operator.attrgetter("_case._global_econ")(self)['ProjectTime'] )
    projectLife += 1

    signal = alpha['signals']
    set_scenarios = alpha['sets']['synth_scenarios']
    set_years = alpha['sets']['projYearsRange'] if self._testMode else alpha['sets']['synth_years']
    set_days  = alpha['sets']['synth_days']
    set_time  = alpha['sets']['synth_hours']

    # time indeces for HERON/TEAL
    n_hours = len(set_time)
    n_days  = len(set_days)
    n_scenarios = len(set_scenarios)
    n_hours_per_year = n_hours * n_days # sometimes number of days refers to clusters < 365

    # plus 1 to year term to allow for 0 recurring costs during build year
    reshaped_alpha = np.zeros([n_scenarios, projectLife, n_hours_per_year])

    for real in set_scenarios:
      # it necessary to have alpha be [real, year, hour] instead of [real, year, cluster, hour]
      realized_alpha = [[signal[real][y][d][t] \
                                    for d in set_days
                                      for t in set_time]
                                        for y in set_years] #shape here is [year, hour]
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
      @In: kwargs: extra arguments for flowsheet parameters
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
      @ In, scenario, Pyomo model for given scenario
      @ In, set_time, list of hours in day
      @ In, set_days, list of days/clusters per year
      @ In, set_years, list of years per scenario
      @ Out, None
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
      Adding non-anticipativity constraints, ensuring that all capacity variables are the same
      for all scenarios.
      @ In, None
      @ Out, None
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
      Adding objective function using TEAL metrics.
      @ In, None
      @ Out, None
    """
    # scenario probability weights
    weights = self._dmdl.weights_scenarios

    # pyomo expression for full metric wtih scenario weights applied
    Metric = np.sum( weights[n]*scenario['NPV'] for n, scenario in enumerate(self._metrics) )

    # set objective
    self._dmdl.obj = pyo.Objective(expr=Metric, sense=pyo.maximize)

  def _solveDispatchesModel(self):
    """
      Solve the DISPATCHES Pyomo model.
      @ In, None
      @ Out, None
    """
    # Define the solver object. Using the default solver: IPOPT
    solver = get_solver()

    # Solve the optimization problem
    self._results = solver.solve(self._dmdl, tee=True)

  def _exportResults(self):
    """
      Print and export results from optimization solution.
      @ In, None
      @ Out, None
    """
    mdl = self._dmdl
    case_name = self._case.name

    # optimal NPV and variable values
    opt_NPV = pyo.value(mdl.obj) # given in $s
    opt_PEM = pyo.value(mdl.pem_capacity) # already in kW
    opt_H2Tank = pyo.value(mdl.tank_capacity) * 2.016e-3 # convert to kg
    opt_H2Turb = pyo.value(mdl.h2_turbine_capacity) * 1e-3 # convert to kW

    print(f'Optimal NPV is ------------------- $B {opt_NPV * 1e-9} ')
    print(f'--- Optimal PEM capacity is        {opt_PEM  * 1e-3} MW')
    print(f'--- Optimal H2 Tank capacity is    {opt_H2Tank} kg')
    print(f'--- Optimal H2 Turbine capacity is {opt_H2Turb * 1e-3} MW')

    # storing and outputting optimal values to csv
    columns = []
    values  = []
    # optimal objective / metric
    columns.append('Expected NPV')
    values.append(opt_NPV)
    # optimal capacities
    columns.extend(['PEM Size', 'H2 Tank Size', 'H2 Turbine Size'])
    values.extend([opt_PEM, opt_H2Tank, opt_H2Turb])
    # outputting to csv
    output_data = pd.DataFrame([values], columns=columns)
    output_data.to_csv(f'opt_solution__{case_name}.csv')

  def run(self):
    """
      Runs the DISPATCHES workflow
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

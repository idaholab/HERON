#!/usr/bin/env python
# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  New HERON workflow for setting up and running DISPATCHES cases
  HEron Runs Dispatches (HERD)
"""
import os, sys
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import _utils as hutils
import numpy as np
from functools import partial
from itertools import compress
path_to_raven = hutils.get_raven_loc()
sys.path.append(os.path.abspath(os.path.join(path_to_raven, 'scripts')))
sys.path.append(os.path.abspath(os.path.join(path_to_raven, 'plugins')))
sys.path.append(path_to_raven)
from TEAL.src import CashFlows
from TEAL.src import main as RunCashFlow
from HERON.src.Moped import MOPED


dispatches_model_component_meta={
  "Nuclear-Hydrogen IES: H2 Production, Storage, and Combustion": {
    "npp":{ # currently, this only produces baseload electricity
        "Produces": 'electricity',
        "Consumes": {},
    },
    "pem":{ # will require some transfer function
        "Produces": 'hydrogen',
        "Consumes": 'electricity',
    },
    "h2tank":{
        "Stores":   'hydrogen',
        "Consumes": {},
    },
    "h2turbine":{ # TODO: technically also consumes air, will need to revisit
        "Produces": 'electricity',
        "Consumes": 'hydrogen',
    },
    "electricity_market":{
        "Demands":  'electricity',
        "Consumes": {},
    },
    "h2_market":{
        "Demands":  'hydrogen',
        "Consumes": {},
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
      for prod in getattr(comp, "_produces"): # NOTE Cannot handle components that produce multiple things
        self.getComponentActionMeta(comp, prod, "Produces")
      for sto in getattr(comp, "_stores"):
        self.getComponentActionMeta(comp, sto, "Stores")
      for dem in getattr(comp, "_demands"): # NOTE Cannot handle components that demand multiple things
        self.getComponentActionMeta(comp, dem, "Demands")

  def getComponentActionMeta(self, comp, action, action_type=None):
    """
      Checks the capacity type, dispatch type, and resources involved for each component
      to build component_meta. Repurposed from MOPED
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

    # saving resource under action type, e.g. "Produces": "electricity"
    self._component_meta[comp.name][action_type] = resource
    self._component_meta[comp.name]['Consumes'] = {}
    self._component_meta[comp.name]['Dispatch'] = getattr(action, "_dispatchable")

    # save optimization parameters
    if mode in ('OptBounds', 'FixedValue'):
      self.verbosityPrint(f'|Building pyomo capacity {mode} for {comp.name}|')
      self._component_meta[comp.name][mode] = getattr(value, "_parametric")

    # sample synthetic histories
    elif mode == 'SyntheticHistory':
      self.verbosityPrint(f'|Building pyomo parameter with synthetic histories for {comp.name}|')
      synthHist = self.loadSyntheticHistory( getattr(value, "_var_name") ) # runs external ROM load
      self._component_meta[comp.name][mode] = synthHist

    # cannot do sweep values yet
    elif mode == 'SweepValues': # TODO Add capability to handle sweepvalues, maybe multiple pyo.Params?
      raise IOError('MOPED does not currently support sweep values option')

    # NOTE not all producers consume
    # TODO should we handle transfer functions here?
    if consumes:
      for con in getattr(action, "_consumes"):
        self._component_meta[comp.name]['Consumes'][con] = getattr(action, "_transfer")

  def loadSyntheticHistory(self, signal):
    """
      Loads synthetic history for a specified signal,
      also sets yearly hours and pyomo indexing sets.
      Calls the parent method and restructures dictionary
      to match DISPATCHES format.
      @ In, signal, string, name of signal to sample
      @ Out, synthetic_data, dict, contains data from evaluated ROM
    """
    # calling parent method for loading synthetic history
    synthHist = super().loadSyntheticHistory(signal)

    # extracting inner data array shapes
    realizations = list( synthHist.keys() )

    # NOTE: assuming all realizations have same array shape
    n_years, n_clusters, n_hours = synthHist[realizations[0]].shape

    # some time sets to describe synthetic histories
    set_scenarios = range(len(realizations))
    set_years = synthHist['years']
    set_days  = range(1, n_clusters + 1) # to appease Pyomo, indexing starts at 1
    set_time  = range(1, n_hours + 1)    # to appease Pyomo, indexing starts at 1

    # double check years
    if len(set_years) != n_years:
      raise IOError("Discrepancy in number of years within Synthetic History")

    # restructure the synthetic history dictionary to match DISPATCHES
    newHist = {}
    for key, data in synthHist.items():
      # assuming the keys are in format "Realization_i"
      if "Realization" in key:
        # realizations known as scenarios in DISPATCHES, index starting at 0
        k = int( key.split('_')[-1] )
        # years indexed by integer year (2020, etc.)
        # clusters and hours indexed starting at 1
        newHist[set_scenarios[k-1]] = {year: {cluster: {hour:  data[y, cluster-1, hour-1]
                                                      for hour in set_time}
                                            for cluster in set_days}
                                    for y, year in enumerate(set_years)}

    # save set time data for use within DISPATCHES
    newHist["indeces"] = {}
    newHist["indeces"]["set_scenarios"] = set_scenarios
    newHist["indeces"]["set_years"] = set_years
    newHist["indeces"]["set_days"]  = set_days
    newHist["indeces"]["set_time"]  = set_time
    return newHist

  def _checkDispatchesCompatibility(self):
    """
      Checks HERON components to match compatibility with available
      DISPATCHES flowsheets.
    """
    heron_comp_list = list( self._component_meta.keys() ) # current list of HERON components
    self.verbosityPrint(f'|Checking compatibility between HERON and available DISPATCHES cases|')

    # check that HERON input file contains all components needed to run DISPATCHES case
    # using naming convention: d___ corresponds to DISPATCHES, h___ corresponds to HERON
    for dName, dModel in dispatches_model_component_meta.items():
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
            mismatched_actions.append(hResource != dResource )
          else:
            mismatched_actions.append(hAction != dResource )

        if sum(mismatched_actions) > 0:
          message = f'Attributes of HERON Component {dComp} do not match DISPATCHES case: '
          message += ', '.join( list(compress(dispatches_actions_list, mismatched_actions)) )
          raise IOError(message)

      self.verbosityPrint(f'|HERON Case is compatible with {dName} DISPATCHES Model|')

  def run(self):
    """
      Runs the workflow
      @ In, None
      @ Out, None
    """
    self.buildEconSettings()
    self.buildComponentMeta()
    self.collectResources()
    self._checkDispatchesCompatibility()

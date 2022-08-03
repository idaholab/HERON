# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  New HERON workflow for setting up and running DISPATCHES cases
  HEron Runs Dispatches (HERD)
"""
import os
import sys
from itertools import compress
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import _utils as hutils

# Nuclear flowsheet function imports
from dispatches.models.nuclear_case.flowsheets.nuclear_flowsheet import build_ne_flowsheet
from dispatches.models.nuclear_case.flowsheets.nuclear_flowsheet import fix_dof_and_initialize

# Import function for the construction of the multiperiod model
from dispatches.models.nuclear_case.flowsheets.multiperiod import build_multiperiod_design

# append path with RAVEN location
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
    self._dmdl = None # Pyomo model specific to DISPATCHES (different from _m)

  # def buildCashflowMeta(self):
  #   """
  #     Builds cashflow meta used in cashflow component construction
  #     @ In, None
  #     @ Out, None
  #   """
  #   # NOTE MOPED version assumes that each component can only have one cap, yearly, and repeating
  #   #   here we keep all cashflows even if there are multipler for cap, yearly, or repeating
  #   for comp in self._components:
  #     self.verbosityPrint(f'Retrieving cashflow information for {comp.name}')

  #     # setting up empty dict, getting Lifetime in years
  #     econ = getattr(comp, "_economics")
  #     self._cf_meta[comp.name] = {}
  #     self._cf_meta[comp.name]['Lifetime'] = getattr(econ, "_lifetime")

  #     # loop through all cash flows for given component
  #     cashflows = getattr(econ, "_cash_flows")
  #     for cf in cashflows:
  #       # Using reference prices for cashflows (getting annoying warnings about protected members)
  #       alpha    = getattr(cf, "_alpha")
  #       alpha_vp = getattr(alpha, "_vp") # valued param object
  #       alpha_v  = getattr(alpha_vp, "_parametric") # alpha value
  #       driver   = getattr(cf, "_driver")
  #       multiplier = getattr(driver, "_multiplier") # default of 1
  #       multiplier = 1 if multiplier is None else multiplier
  #       value = multiplier * alpha_v

  #       # getting cashflow type, making sure it is within accepted types
  #       cf_type = getattr(cf, "_type")
  #       if cf_type not in ['one-time', 'yearly', 'repeating']:
  #         raise IOError("Cashflow type not currently supported in HERD")

  #       # deviation from MOPED, storing type and given name
  #       self._cf_meta[comp.name][f'{cf_type}|{cf.name}'] = value

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
    set_years = list(synthHist['years'])
    set_days  = range(1, n_clusters + 1) # to appease Pyomo, indexing starts at 1
    set_time  = range(1, n_hours + 1)    # to appease Pyomo, indexing starts at 1

    # double check years
    if len(set_years) != n_years:
      raise IOError("Discrepancy in number of years within Synthetic History")

    # restructure the synthetic history dictionary to match DISPATCHES
    newHist = {}
    newHist['signals'] = {}
    for key, data in synthHist.items():
      # assuming the keys are in format "Realization_i"
      if "Realization" in key:
        # realizations known as scenarios in DISPATCHES, index starting at 0
        k = int( key.split('_')[-1] )
        # years indexed by integer year (2020, etc.)
        # clusters and hours indexed starting at 1
        newHist['signals'][set_scenarios[k-1]] = {year: {cluster: {hour:  data[y, cluster-1, hour-1]
                                                                  for hour in set_time}
                                                        for cluster in set_days}
                                                  for y, year in enumerate(set_years)}

    # save set time data for use within DISPATCHES
    newHist["sets"] = {}
    newHist["sets"]["set_scenarios"] = list(set_scenarios) # DISPATCHES wants this as a list
    newHist["sets"]["set_years"] = set_years # DISPATCHES wants this as a list
    newHist["sets"]["set_days"]  = set_days  # DISPATCHES wants this as a range
    newHist["sets"]["set_time"]  = set_time  # DISPATCHES wants this as a range

    #TODO: need weights_days - how many days does each cluster represent?
    if n_clusters == 2:
      tmp_weights = [183, 182] # just guessing for now, need this info from ARMA model?
      newHist["weights_days"]  = {year: {cluster: tmp_weights[cluster-1]
                                      for cluster in set_days}
                                for year in set_years}
    else:
      raise IOError("HERD not set up for more than 2 clusters yet.")

    return newHist

  def _checkDispatchesCompatibility(self):
    """
      Checks HERON components to match compatibility with available
      DISPATCHES flowsheets.
    """
    # TODO: check for financial params/inputs?
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

  def _createDispatchesPyomoModel(self):
    """
      Create new DISPATCHES Pyomo Model with available data
    """

    self._dmdl = pyo.ConcreteModel(name=self._case.name)

    market_synthetic_history = self._component_meta['electricity_market']['SyntheticHistory']

    # transferring information on Sets
    sets = market_synthetic_history['sets']
    self._dmdl.set_time  = sets['set_time']
    self._dmdl.set_days  = sets['set_days']
    self._dmdl.set_years = sets['set_years']
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
    # temporary object pointing to model
    dmdl = self._dmdl

    mdl_flowsheet = build_ne_flowsheet # pointing to the imported DISPATCHES nuclear flowsheet
    mdl_init = fix_dof_and_initialize # pointing to the imported DISPATCHES fix/init method
    mdl_unfix = self.unfix_dof # we add a method to unfix certain DoFs based on DISPATCHES jupyter notebooks

    # NOTE: within the build process, a tmp JSON file is created in wdir... ugh.
    build_multiperiod_design(dmdl,
                         flowsheet=mdl_flowsheet,
                         initialization=mdl_init,
                         unfix_dof=mdl_unfix,
                         multiple_days=True,
                         multiyear=True,
                         stochastic=True,
                         verbose=False)

    for s in dmdl.set_scenarios:
      # Build the connecting constraints
      self.build_connecting_constraints(dmdl.scenario[s],
                                  set_time=dmdl.set_time,
                                  set_days=dmdl.set_days,
                                  set_years=dmdl.set_years)

      # Append cash flow expressions
      self.append_costs_and_revenue(dmdl.scenario[s],
                              ps=dmdl,
                              LMP=dmdl.LMP[s])

    #   # Hydrogen demand constraint.
    #   # Divide the RHS by the molecular mass to convert kg/s to mol/s
    #   scenario = dmdl.scenario[s]
    #   @scenario.Constraint(dmdl.set_time, dmdl.set_days, dmdl.set_years)
    #   def hydrogen_demand_constraint(blk, t, d, y):
    #       return blk.period[t, d, y].fs.h2_tank.outlet_to_pipeline.flow_mol[0] <= dmdl.h2_demand / 2.016e-3

  def run(self):
    """
      Runs the workflow
      @ In, None
      @ Out, None
    """
    # original workflow from MOPED
    self.buildEconSettings()  # MOPED method
    self.buildComponentMeta() # overloaded method
    self.buildCashflowMeta()  # overloaded method
    self.collectResources()   # MOPED method (TODO: needed?)

    # new workflow for DISPATCHES
    self._checkDispatchesCompatibility()
    self._createDispatchesPyomoModel()
    self._buildDispatchesModel()
    # self._solveDispatchesModel()

    # now we need to import DISPATCHES
    #  X. load in the synthetic histories to DISPATCHES
    #  X. create weights
    #  X. create overall Pyomo model
    #  X. set global parameters from HERON
    #  X. build DISPATCHES multiperiod
    #      X. point to existing flowsheet
    #      X. point to existing init/fix method
    #      X. **NEW** overload unfix method??
    #  6. run a loop over scenarios
    #      X. build constraints (**NEW** import some params from HERON??) per scenario
    #      b. run append_costs
    #           i. use Pyomo expressions to create TEAL cashflows, particularly npvs per scenario
    #           ii. loop through dispatches_meta_dict?
    #                - create capex pyomo using correct drivers
    #  7. add non-anticipativity constraints
    #  8. add objective (sum up npvs)
    #  9. run solver from idaes


  ############################
  # DISPATCHES methods

  def unfix_dof(self, m, **kwargs):
    """
    This function unfixes a few degrees of freedom for optimization.
    This particular method is taken from the DISPATCHES jupyter notebook
    found in "dispatches/dispatches/models/nuclear_case/flowsheets"
    titled "multiperiod_design_pricetaker"
    """
    # Set defaults in case options are not passed to the function
    options = kwargs.get("options", {})
    air_h2_ratio = options.get("air_h2_ratio", 10.76)

    # Unfix the electricity split in the electrical splitter
    m.fs.np_power_split.split_fraction["np_to_grid", 0].unfix()

    # Unfix the holdup_previous and outflow variables
    m.fs.h2_tank.tank_holdup_previous.unfix()
    m.fs.h2_tank.outlet_to_turbine.flow_mol.unfix()
    m.fs.h2_tank.outlet_to_pipeline.flow_mol.unfix()

    # Unfix the flowrate of air to the mixer
    m.fs.mixer.air_feed.flow_mol.unfix()

    # Add a constraint to maintain the air to hydrogen flow ratio
    m.fs.mixer.air_h2_ratio = pyo.Constraint(
                expr=m.fs.mixer.air_feed.flow_mol[0] ==
                      air_h2_ratio * m.fs.mixer.hydrogen_feed.flow_mol[0])

    # Set bounds on variables. A small non-zero value is set as the lower
    # bound on molar flowrates to avoid convergence issues
    m.fs.pem.outlet.flow_mol[0].setlb(0.001)

    m.fs.h2_tank.inlet.flow_mol[0].setlb(0.001)
    m.fs.h2_tank.outlet_to_turbine.flow_mol[0].setlb(0.001)
    m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].setlb(0.001)

    m.fs.translator.inlet.flow_mol[0].setlb(0.001)
    m.fs.translator.outlet.flow_mol[0].setlb(0.001)

    m.fs.mixer.hydrogen_feed.flow_mol[0].setlb(0.001)

  def build_connecting_constraints(self, m, set_time, set_days, set_years):
    """
    This function declares the first-stage variables or design decisions,
    adds constraints that ensure that the operational variables never exceed their
    design values, and adds constraints connecting variables at t - 1 and t
    """
    # Declare first-stage variables (Design decisions)
    m.pem_capacity = pyo.Var(within=pyo.NonNegativeReals,
                          doc="Maximum capacity of the PEM electrolyzer (in kW)")
    m.tank_capacity = pyo.Var(within=pyo.NonNegativeReals,
                          doc="Maximum holdup of the tank (in mol)")
    m.h2_turbine_capacity = pyo.Var(within=pyo.NonNegativeReals,
                                doc="Maximum power output from the turbine (in W)")

    # Ensure that the electricity to the PEM elctrolyzer does not exceed the PEM capacity
    @m.Constraint(set_time, set_days, set_years)
    def pem_capacity_constraint(blk, t, d, y):
      return blk.period[t, d, y].fs.pem.electricity[0] <= m.pem_capacity

    # Ensure that the final tank holdup does not exceed the tank capacity
    @m.Constraint(set_time, set_days, set_years)
    def tank_capacity_constraint(blk, t, d, y):
      return blk.period[t, d, y].fs.h2_tank.tank_holdup[0] <= m.tank_capacity

    # Ensure that the power generated by the turbine does not exceed the turbine capacity
    @m.Constraint(set_time, set_days, set_years)
    def turbine_capacity_constraint(blk, t, d, y):
      return (
          - blk.period[t, d, y].fs.h2_turbine.turbine.work_mechanical[0]
          - blk.period[t, d, y].fs.h2_turbine.compressor.work_mechanical[0] <=
          m.h2_turbine_capacity
      )

    # Connect the initial tank holdup at time t with the final tank holdup at time t - 1
    @m.Constraint(set_time, set_days, set_years)
    def tank_holdup_constraints(blk, t, d, y):
      if t == 1:
        # Each day begins with an empty tank
        return (
          blk.period[t, d, y].fs.h2_tank.tank_holdup_previous[0] == 0
        )
      else:
        # Initial holdup at time t = final holdup at time t - 1
        return (
          blk.period[t, d, y].fs.h2_tank.tank_holdup_previous[0] ==
          blk.period[t - 1, d, y].fs.h2_tank.tank_holdup[0]
        )

  def append_costs_and_revenue(self, m, ps, LMP):
    """
    ps: Object containing information on sets and parameters
    LMP: Dictionary containing the LMP data
    """

    set_time = ps.set_time             # Set of hours
    set_days = ps.set_days             # Set of days/clusters
    set_years = ps.set_years           # Set of years
    weights_days = ps.weights_days     # Weights associated with each cluster

    h2_sp = ps.h2_price                # Selling price of hydrogen
    plant_life = ps.plant_life         # Plant lifetime
    tax_rate = ps.tax_rate             # Corporate tax rate
    discount_rate = ps.discount_rate   # Discount rate

    years_vec = [y - set_years[0] + 1 for y in set_years]
    years_vec.append(plant_life + 1)
    weights_years = {y: sum(1 / (1 + discount_rate) ** i
                            for i in range(years_vec[j], years_vec[j + 1]))
                      for j, y in enumerate(set_years)}

      # # PEM CAPEX: $1630/kWh and pem_capacity is in kW,
      # # Tank CAPEX: $29/kWh, the LHV of hydrogen is 33.3 kWh/kg,
      # # the molecular mass of hydrogen is 2.016e-3 kg/mol and
      # # tank_capacity is in moles
      # # Turbine CAPEX: $947/kWh and turbine_capacity is in W
      # m.capex = Expression(
      #     expr=(1630 * m.pem_capacity +
      #           (29 * 33.3 * 2.016e-3) * m.tank_capacity +
      #           (947 / 1000) * m.h2_turbine_capacity),
      #     doc="Total capital cost (in USD)"
      # )

      # # Fixed O&M of PEM: $47.9/kW
      # # Fixed O&M of turbine: $7/kW
      # @m.Expression(set_years,
      #               doc="Fixed O&M cost per year (in USD)")
      # def fixed_om_cost(blk, y):
      #     return (
      #         47.9 * m.pem_capacity + 7e-3 * m.h2_turbine_capacity
      #     )

      # # Variable O&M: PEM: $1.3/MWh and turbine: $4.25/MWh
      # @m.Expression(set_years,
      #               doc="Total variable O&M cost per year (in USD)")
      # def variable_om_cost(blk, y):
      #     return (
      #         (1.3 * 1e-3) * sum(weights_days[y][d] * blk.period[t, d, y].fs.pem.electricity[0]
      #                            for t in set_time for d in set_days) +
      #         (4.25 * 1e-6) * sum(weights_days[y][d] * (
      #                             - blk.period[t, d, y].fs.h2_turbine.turbine.work_mechanical[0]
      #                             - blk.period[t, d, y].fs.h2_turbine.compressor.work_mechanical[0])
      #                             for t in set_time for d in set_days)
      #     )

      # @m.Expression(set_years,
      #               doc="Revenue generated by selling electricity per year (in USD)")
      # def electricity_revenue(blk, y):
      #     return (
      #         sum(weights_days[y][d] * LMP[y][d][t] *
      #             (blk.period[t, d, y].fs.np_power_split.np_to_grid_port.electricity[0] * 1e-3 -
      #              blk.period[t, d, y].fs.h2_turbine.turbine.work_mechanical[0] * 1e-6 -
      #              blk.period[t, d, y].fs.h2_turbine.compressor.work_mechanical[0] * 1e-6)
      #             for t in set_time for d in set_days)
      #     )

      # @m.Expression(set_years,
      #               doc="Revenue generated by selling hydrogen per year (in USD)")
      # def h2_revenue(blk, y):
      #     return (
      #         h2_sp * 2.016e-3 * 3600 *
      #         sum(weights_days[y][d] *
      #             blk.period[t, d, y].fs.h2_tank.outlet_to_pipeline.flow_mol[0]
      #             for t in set_time for d in set_days)
      #     )

      # @m.Expression(set_years,
      #               doc="Depreciation value per year (in USD)")
      # def depreciation(blk, y):
      #     return (
      #         blk.capex / plant_life
      #     )

      # @m.Expression(set_years,
      #               doc="Net profit per year (in USD)")
      # def net_profit(blk, y):
      #     return (
      #         blk.depreciation[y] + (1 - tax_rate) * (+ blk.h2_revenue[y]
      #                                                 + blk.electricity_revenue[y]
      #                                                 - blk.fixed_om_cost[y]
      #                                                 - blk.variable_om_cost[y]
      #                                                 - blk.depreciation[y])
      #     )

      # m.npv = Expression(
      #     expr=sum(weights_years[y] * m.net_profit[y] for y in set_years) - m.capex,
      #     doc="Net present value (in USD)"
      # )


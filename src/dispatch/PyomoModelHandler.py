# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  This module constructs the dispatch optimization model used by HERON.
"""
import os
import sys
try:
  import dove.core as dv
except ImportError:
  # TODO: temporary solution that works when DOVE and HERON dirs share a parent
  sys.path.append(
    os.path.abspath(
      os.path.join(__file__, os.pardir, os.pardir, os.pardir, os.pardir, 'DOVE', 'src')
    )
  )
  import dove.core as dv

import numpy as np
import pyomo.environ as pyo
from copy import deepcopy

from . import PyomoRuleLibrary as prl
from . import putils
from .DispatchState import PyomoState

class PyomoModelHandler:
  """
    Class for constructing the pyomo model, populate with objective/constraints, and evaluate.
  """

  _eps = 1e-9

  def __init__(self, time, time_offset, case, components, resources, initial_storage, meta) -> None:
    """
      Initializes a PyomoModelHandler instance.
      @ In, time, np.array(float), time values to evaluate; may be length 1 or longer
      @ In, time_offset, int, optional, increase time index tracker by this value if provided
      @ In, case, HERON Case, case to evaluate
      @ In, components, list, HERON components to evaluate
      @ In, resources, list, HERON resources to evaluate
      @ In, initial_storage, dict, initial storage levels
      @ In, meta, dict, additional state information
      @ Out, None
    """
    self.time = time
    self.time_offset = time_offset
    self.case = case
    self.components = components
    self.resources = resources
    self.initial_storage = initial_storage
    self.resource_index_map = meta["HERON"]["resource_indexer"]
    self.meta = meta
    self.model = self.build_model()


  def build_model(self):
    """
      Construct the skeleton of the pyomo model.
      @ In, None
      @ Out, model, pyo.ConcreteModel, model
    """
    model = pyo.ConcreteModel()
    C = np.arange(0, len(self.components), dtype=int) # indexes component
    R = np.arange(0, len(self.resources), dtype=int) # indexes resources
    T = np.arange(0, len(self.time), dtype=int) # indexes resources
    model.C = pyo.Set(initialize=C)
    model.R = pyo.Set(initialize=R)
    model.T = pyo.Set(initialize=T)
    model.Times = self.time
    model.time_offset = self.time_offset
    # maps the resource to its index WITHIN APPLICABLE components (sparse matrix)
    # e.g. component: {resource: local index}, ... etc}
    model.resource_index_map = self.meta['HERON']['resource_indexer']
    # properties
    model.Case = self.case
    model.Components = self.components
    model.Activity = PyomoState()
    model.Activity.initialize(model.Components, model.resource_index_map, model.Times, model)
    return model


  def populate_model(self):
    """
      Populate the pyomo model with generated objectives/contraints.
      @ In, None
      @ Out, None
    """
    context = deepcopy(self.meta)

    dove_res_map = self._create_dove_resources(context)
    dove_comp_list = []
    for comp in self.components:
      caps = []  # Time dependent capacity values
      mins = []  # Time dependent minimum values
      cf_map = {} # dict keyed by cashflow names with values that are dicts containing economic info
      for t in range(len(self.model.Times)):
        # update time index in meta for capacity/minimum evaluation
        context['HERON']['time_index'] = t + self.model.time_offset
        cap_val = comp.get_capacity(context)[0][comp.get_capacity_var()] # get capacity for this component
        caps.append(cap_val)
        mins.append(comp.get_minimum(context)[0][comp.get_capacity_var()]) # get minimum for this component
        # We have to spoof activity to get the other cashflow params
        recurring_cfs = [cf for cf in comp.get_cashflows() if cf.get_type() == 'repeating' and cf.get_period() != "year"]
        for cf in recurring_cfs:
          if cf.name not in cf_map.keys():
            cf_map[cf.name] = {"alphas": [], "dprimes": [], "scaling_factors": [], "d_multipliers": [], "costs": []}  # costs only used for sign
          context["HERON"]["activity"] = {cf.get_driver()._vp._tracking_var: {comp.get_capacity_var(): -1}} # NOTE: setting to 1 allows us to get the multiplier
          params = cf.calculate_params(context)
          cf_map[cf.name]["alphas"].append(params["alpha"])
          cf_map[cf.name]["dprimes"].append(params["ref_driver"])
          cf_map[cf.name]["scaling_factors"].append(params["scaling"])
          cf_map[cf.name]["d_multipliers"].append(params["driver"])
          cf_map[cf.name]["costs"].append(params["cost"])
          mult_target = cf.is_mult_target()

      if comp.get_interaction().get_transfer() is not None:
        comp._coeffs = comp.get_interaction().get_transfer().get_coefficients()
      comp._capacity_vector_t = caps
      comp._minimum_vector_t = mins
      comp._cfs = cf_map
      comp._r = comp.get_capacity_var()

      if comp.get_interaction().is_type("Storage"):
        comp._initial_storage = comp.get_interaction().get_initial_level(context)
        comp._max_charge, comp._max_discharge = comp.get_interaction().get_charge_rate_limits(context)
        comp._periodic_level = comp.get_interaction().apply_periodic_level
        # if comp.is_governed():
        #   comp._activity = comp.get_interaction().get_strategy().evaluate(self.meta)[0]['level']
        # TODO: could we handle this?

      dove_comp = self._create_dove_component(comp, dove_res_map)
      dove_comp_list.append(dove_comp)

    dove_system = dv.System(
      components=dove_comp_list,
      resources=list(dove_res_map.values()),
      dispatch_window=np.arange(0, len(self.time), dtype=int)
    )
    dispatch_results = dove_system.solve(model="price_taker")
    print(dispatch_results)

  def _create_dove_resources(self, context):
    heron_res_map = context['HERON']['resource_indexer']
    dove_res_map = {}
    for indexer_per_comp in heron_res_map.values():
      for r_name in indexer_per_comp.keys():
        if r_name not in dove_res_map.keys():
          dove_res_map[r_name] = dv.Resource(name=r_name)
    return dove_res_map

  def _create_dove_component(self, heron_comp, dove_res_map):
    interaction = heron_comp.get_interaction()
    if interaction.is_type("Producer"):
      if interaction.get_transfer() is None:
        dove_comp = self._create_dove_source(heron_comp, dove_res_map)
      else:
        dove_comp = self._create_dove_converter(heron_comp, dove_res_map)
    elif interaction.is_type("Storage"):
      dove_comp = self._create_dove_storage(heron_comp, dove_res_map)
    elif interaction.is_type("Demand"):
      dove_comp = self._create_dove_sink(heron_comp, dove_res_map)
    else:
      raise IOError(f"{heron_comp.name}: Unrecognized interaction type. Please use 'Producer', 'Storage', or 'Demand'.")

    return dove_comp

  def _create_dove_source(self, heron_comp, dove_res_map):
    init_kwargs = {}
    init_kwargs["name"] = heron_comp.name
    init_kwargs["installed_capacity"] = max(heron_comp._capacity_vector_t)
    if not all(cap_val == init_kwargs["installed_capacity"] for cap_val in heron_comp._capacity_vector_t):
      init_kwargs["capacity_factor"] = [
        cap_val / init_kwargs["installed_capacity"] for cap_val in heron_comp._capacity_vector_t
      ]
    if heron_comp.is_dispatchable() == "fixed":
      init_kwargs["flexibility"] = "fixed"
    elif init_kwargs["installed_capacity"] > 0:
      # The minimum would be overridden by fixed flexibility, so only worry about it if it's flexible
      init_kwargs["min_capacity_factor"] = [
        min_val / init_kwargs["installed_capacity"] for min_val in heron_comp._minimum_vector_t
      ]
    if heron_comp._cfs:
      init_kwargs["cashflows"] = []
      for cf_name, cf_data in heron_comp._cfs.items():
        init_kwargs["cashflows"].append(self._create_dove_cashflow(cf_name, cf_data, 1))
    init_kwargs["produces"] = dove_res_map[heron_comp._r]

    # TODO: HERON supports ramp_limits and ramp_freq for producers; DOVE does not
    # Need to fix incompatibility issue

    return dv.Source(**init_kwargs)

  def _create_dove_sink(self, heron_comp, dove_res_map):
    init_kwargs = {}
    init_kwargs["name"] = heron_comp.name
    init_kwargs["demand_profile"] = [abs(cap_at_t) for cap_at_t in heron_comp._capacity_vector_t]
    if heron_comp.is_dispatchable() == "fixed":
      init_kwargs["flexibility"] = "fixed"
    else:
      # The minimum would be overridden by fixed flexibility, so only worry about it if it's flexible
      init_kwargs["min_demand_profile"] = [abs(min_val) for min_val in heron_comp._minimum_vector_t]
    if heron_comp._cfs:
      init_kwargs["cashflows"] = []
      for cf_name, cf_data in heron_comp._cfs.items():
        init_kwargs["cashflows"].append(self._create_dove_cashflow(cf_name, cf_data, -1))
    init_kwargs["consumes"] = dove_res_map[heron_comp._r]

    return dv.Sink(**init_kwargs)

  def _create_dove_converter(self, heron_comp, dove_res_map):
    init_kwargs = {}
    init_kwargs["name"] = heron_comp.name

    init_kwargs["installed_capacity"] = max(abs(cap_val) for cap_val in heron_comp._capacity_vector_t)
    if not all(cap_val == heron_comp._capacity_vector_t[0] for cap_val in heron_comp._capacity_vector_t):
      init_kwargs["capacity_factor"] = [
        abs(cap_val) / init_kwargs["installed_capacity"] for cap_val in heron_comp._capacity_vector_t
      ]
    if heron_comp.is_dispatchable() == "fixed":
      init_kwargs["flexibility"] = "fixed"
    elif init_kwargs["installed_capacity"] > 0:
      # The minimum would be overridden by fixed flexibility, so only worry about it if it's flexible
      init_kwargs["min_capacity_factor"] = [
        abs(min_val) / init_kwargs["installed_capacity"] for min_val in heron_comp._minimum_vector_t
      ]

    init_kwargs["consumes"] = [dove_res_map[res_name] for res_name in list(heron_comp.get_inputs())]
    init_kwargs["produces"] = [dove_res_map[res_name] for res_name in list(heron_comp.get_outputs())]
    init_kwargs["capacity_resource"] = dove_res_map[heron_comp._r]

    match heron_comp.get_interaction().get_transfer().type:
      case "Ratio":
        tf_inputs = {dove_res: abs(heron_comp._coeffs[dove_res.name]) for dove_res in init_kwargs["consumes"]}
        tf_outputs = {dove_res: abs(heron_comp._coeffs[dove_res.name]) for dove_res in init_kwargs["produces"]}
        init_kwargs["transfer_fn"] = dv.RatioTransfer(input_resources=tf_inputs, output_resources=tf_outputs)
      case "Polynomial":
        # TODO
        # HERON has everything on the lhs of the equation (0 on the right)
        # DOVE is incompatible as is (it has the output on the right)
        raise NotImplementedError(f"{heron_comp.name}: This version of HERON does not support Polynomial transfer functions.")
      case _:
        raise IOError(f"{heron_comp.name}: Unrecognized transfer function type.")

    if heron_comp.ramp_limit is not None:
      init_kwargs["ramp_limit"] = heron_comp.ramp_limit()
    if heron_comp.ramp_freq is not None:
      init_kwargs["ramp_freq"] = heron_comp.ramp_freq()

    if heron_comp._cfs:
      init_kwargs["cashflows"] = []
      for cf_name, cf_data in heron_comp._cfs.items():
        cf_default_sign = 1 if init_kwargs["capacity_resource"] in init_kwargs["produces"] else -1
        init_kwargs["cashflows"].append(self._create_dove_cashflow(cf_name, cf_data, cf_default_sign))

    return dv.Converter(**init_kwargs)

  def _create_dove_storage(self, heron_comp, dove_res_map):
    init_kwargs = {}
    init_kwargs["name"] = heron_comp.name

    init_kwargs["installed_capacity"] = max(abs(cap_val) for cap_val in heron_comp._capacity_vector_t)
    if init_kwargs["installed_capacity"] > 0:
      if not all(cap_val == heron_comp._capacity_vector_t[0] for cap_val in heron_comp._capacity_vector_t):
        init_kwargs["capacity_factor"] = [
          abs(cap_val) / init_kwargs["installed_capacity"] for cap_val in heron_comp._capacity_vector_t
        ]
      init_kwargs["min_capacity_factor"] = [
          abs(min_val) / init_kwargs["installed_capacity"] for min_val in heron_comp._minimum_vector_t
        ]
    if heron_comp.is_dispatchable() == "fixed":
      init_kwargs["flexibility"] = "fixed"
    if heron_comp._cfs:
      init_kwargs["cashflows"] = []
      for cf_name, cf_data in heron_comp._cfs.items():
        init_kwargs["cashflows"].append(self._create_dove_cashflow(cf_name, cf_data, 1))

    init_kwargs["resource"] = dove_res_map[heron_comp._r]
    init_kwargs["rte"] = (heron_comp.get_interaction().get_sqrt_RTE())**2
    if heron_comp._max_charge is not None:
      init_kwargs["max_charge_rate"] = heron_comp._max_charge
    if heron_comp._max_discharge is not None:
      init_kwargs["max_discharge_rate"] = heron_comp._max_discharge
    if heron_comp._periodic_level:
      init_kwargs["periodic_level"] = True
      raise NotImplementedError("DOVE can't optimize the initial stored value yet.")
    else:
      init_kwargs["initial_stored"] = heron_comp._initial_storage / init_kwargs["installed_capacity"]
      init_kwargs["periodic_level"] = False

    return dv.Storage(**init_kwargs)

  def _create_dove_cashflow(self, cf_name, cf_data, default_sign):
    '''
    default_sign: 1 if activity causes revenue from this cashflow; -1 if activity causes expense
    '''
    # HERON:  cost  =                      a(t) * [D(activity) / D'(t)]^x
    #  DOVE: |cost| = | price_profile(t) * a    * [ activity   / D'   ]^x |
    # We need to rearrange the HERON equation to:
    # (1) find the absolute value of the cashflow (we'll handle the sign later)
    # (2) move all t-dependencies into the price_profile term
    # (3) convert D(activity) to activity
    cf_kwargs = {"name": cf_name}
    price_profile = []

    # Handle alpha
    if all(alpha_val == cf_data["alphas"][0] for alpha_val in cf_data["alphas"]):
      # HERON a(t) is constant with time
      cf_kwargs["alpha"] = abs(cf_data["alphas"][0])
    else:
      # HERON a(t) varies with time
      cf_kwargs["alpha"] = 1
      price_profile = [abs(alpha_at_t) for alpha_at_t in cf_data["alphas"]]

    # Handle scalex
    if all(sf_val == cf_data["scaling_factors"][0] for sf_val in cf_data["scaling_factors"]):
      # HERON scaling factor is constant with time
      cf_kwargs["scalex"] = cf_data["scaling_factors"][0]
    else:
      # HERON scaling factor varies with time
      raise ValueError(f"{cf_name}: scaling factor must be constant with respect to time.")

    # Handle dprime
    if all(dprime_val == cf_data["dprimes"][0] for dprime_val in cf_data["dprimes"]):
      # HERON D'(t) is constant with time
      cf_kwargs["dprime"] = abs(cf_data["dprimes"][0])
    else:
      # HERON D'(t) varies with time
      # cost = a * [D/D']^x => cost = D'^[-x] * [a * D^x]
      cf_kwargs["dprime"] = 1.0
      if price_profile:
        # The new price_profile has a max length such that there's available data for itself and D'
        allowed_len = min(len(price_profile), len(cf_data["dprimes"]))
        price_profile = [
          price_profile[t] * abs(cf_data["dprimes"][t])**(-1 * cf_kwargs["scalex"]) for t in range(allowed_len)
        ]
      else:
        price_profile = [
          abs(cf_data["dprimes"][t])**(-1 * cf_kwargs["scalex"]) for t in range(len(cf_data["d_primes"]))
        ]

    # Handle driver
    # TODO: Any chance the Driver isn't activity-based? Would need to consider differently.
    # Need to consider cases where HERON D(activity) = multiplier * activity
    # cost = a * [D/D']^x; D = d_mult*activity => cost = d_mult^x * [a * [activity/D']^x]
    if price_profile:
      # The new price_profile has a max length such that there's available data for itself and d_mult
      allowed_len = min(len(price_profile), len(cf_data["d_multipliers"]))
      price_profile = [
        price_profile[t] * abs(cf_data["d_multipliers"][t])**cf_kwargs["scalex"] for t in range(allowed_len)
      ]
    else:
      price_profile = [
        abs(cf_data["d_multipliers"][t])**cf_kwargs["scalex"] for t in range(len(cf_data["d_multipliers"]))
      ]

    # Set price_profile
    if price_profile:
      cf_kwargs["price_profile"] = price_profile

    if all(cost*default_sign >= 0 for cost in cf_data["costs"]):
      return dv.Revenue(**cf_kwargs)
    elif all(cost*default_sign <= 0 for cost in cf_data["costs"]):
      return dv.Cost(**cf_kwargs)
    else:
      raise ValueError(f"{cf_name}: Sign of cashflow must be either always positive or always negative.")

  # TODO: delete below
  def _process_component(self, component):
    """
      Determine what kind of component this is and process it accordingly.
      @ In, component, HERON Component, component to process
      @ Out, None
    """
    interaction = component.get_interaction()
    if interaction.is_governed():
      self._process_governed_component(component, interaction)
    elif interaction.is_type("Storage"):
      self._create_storage(component)
    else:
      self._create_production(component)


  def _process_governed_component(self, component, interaction):
    """
      Process a component that is governed since it requires special attention.
      @ In, component, HERON Component, component to process
      @ In, interaction, HERON Interaction, interaction to process
      @ Out, None
    """
    self.meta["request"] = {"component": component, "time": self.time}
    if interaction.is_type("Storage"):
      self._process_storage_component(component, interaction)
    else:
      activity = interaction.get_strategy().evaluate(self.meta)[0]['level']
      self._create_production_param(component, activity)


  def _process_storage_component(self, component, interaction):
    """
      Process a storage component.
      @ In, component, HERON Component, component to process
      @ In, interaction, HERON Interaction, interaction to process
    """
    activity = interaction.get_strategy().evaluate(self.meta)[0]["level"]
    self._create_production_param(component, activity, tag="level")
    dt = self.model.Times[1] - self.model.Times[0]
    rte2 = component.get_sqrt_RTE()
    deltas = np.zeros(len(activity))
    deltas[1:] = activity[1:] - activity[:-1]
    deltas[0] = activity[0] - interaction.get_initial_level(self.meta)
    charge = np.where(deltas > 0, -deltas / dt / rte2, 0)
    discharge = np.where(deltas < 0, -deltas / dt * rte2, 0)
    self._create_production_param(component, charge, tag="charge")
    self._create_production_param(component, discharge, tag="discharge")


  def _create_production_limit(self, validation):
    """
      Creates pyomo production constraint given validation errors
      @ In, validation, dict, information from Validator about limit violation
      @ Out, None
    """
    # TODO could validator write a symbolic expression on request? That'd be sweet.
    comp = validation['component']
    resource = validation['resource']
    r = self.model.resource_index_map[comp][resource]
    t = validation['time_index']
    limit = validation['limit']
    limits = np.zeros(len(self.model.Times))
    limits[t] = limit
    limit_type = validation['limit_type']
    prod_name = f'{comp.name}_production'
    rule = lambda mod: prl.prod_limit_rule(prod_name, r, limits, limit_type, t, mod)
    constr = pyo.Constraint(rule=rule)
    counter = 1
    name_template = f'{comp.name}_{resource}_{t}_vld_limit_constr_{{i}}'
    # make sure we get a unique name for this constraint
    name = name_template.format(i=counter)
    while getattr(self.model, name, None) is not None:
      counter += 1
      name = name_template.format(i=counter)
    setattr(self.model, name, constr)
    print(f'DEBUGG added validation constraint "{name}"')


  def _create_production_param(self, comp, values, tag=None):
    """
      Creates production pyomo fixed parameter object for a component
      @ In, comp, HERON Component, component to make production variables for
      @ In, values, np.array(float), values to set for param
      @ In, tag, str, optional, if not None then name will be component_[tag]
      @ Out, prod_name, str, name of production variable
    """
    name = comp.name
    if tag is None:
      tag = 'production'
    # create pyomo indexer for this component's resources
    res_indexer = pyo.Set(initialize=range(len(self.model.resource_index_map[comp])))
    setattr(self.model, f'{name}_res_index_map', res_indexer)
    prod_name = f'{name}_{tag}'
    init = (((0, t), values[t]) for t in self.model.T)
    prod = pyo.Param(res_indexer, self.model.T, initialize=dict(init))
    setattr(self.model, prod_name, prod)
    return prod_name


  def _create_production(self, comp):
    """
      Creates all pyomo variable objects for a non-storage component
      @ In, comp, HERON Component, component to make production variables for
      @ Out, prod_name, str, name of the production variable
    """
    prod_name = self._create_production_variable(comp)
    ## if you cannot set limits directly in the production variable, set separate contraint:
    ## Method 1: set variable bounds directly --> TODO more work needed, but would be nice
    # lower, upper = self._get_prod_bounds(m, comp)
    # limits should be None unless specified, so use "getters" from dictionaries
    # bounds = lambda m, r, t: (lower.get(r, None), upper.get(r, None))
    ## Method 2: set variable bounds directly --> TODO more work needed, but would be nice
    # self._create_capacity(m, comp, prod_name, meta)    # capacity constraints
    # transfer function governs input -> output relationship
    self._create_transfer(comp, prod_name)
    # ramp rates
    if comp.ramp_limit is not None:
      self._create_ramp_limit(comp, prod_name)
    return prod_name


  def _create_production_variable(self, comp, tag=None, add_bounds=True, bounds=None, **kwargs):
    """
      Creates production pyomo variable object for a component
      @ In, comp, HERON Component, component to make production variables for
      @ In, tag, str, optional, if not None then name will be component_[tag]; otherwise "production"
      @ In, add_bounds, bool, optional, if True then determine and set bounds for variable
      @ In, bounds, Iterable[float], optional, custom bounds to set for the variable; ignored if add_bounds=True
      @ In, kwargs, dict, optional, passalong kwargs to pyomo variable
      @ Out, prod_name, str, name of production variable
    """
    if tag is None:
      tag = 'production'
    name = comp.name
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    limit_r = self.resource_index_map[comp][cap_res] # production index of the governing resource
    # create pyomo indexer for this component's resources
    indexer_name = f'{name}_res_index_map'
    indexer = getattr(self.model, indexer_name, None)
    if indexer is None:
      indexer = pyo.Set(initialize=range(len(self.resource_index_map[comp])))
      setattr(self.model, indexer_name, indexer)
    prod_name = f'{name}_{tag}'
    caps, mins = comp._capacity_vector_t, comp._minimum_vector_t
    if min(caps) < 0:
      # quick check that capacities signs are consistent #FIXME: revisit, this is an assumption
      assert max(caps) <= 0, \
        'Capacities are inconsistent: mix of positive and negative values not currently  supported.'
      # we have a unit that's consuming, so we need to flip the variables to be sensible
      mins, caps = caps, mins
      inits = caps
    else:
      inits = mins
    if add_bounds:
      # create bounds based in min, max operation
      bounds = lambda m, r, t: (mins[t] if r == limit_r else None, caps[t] if r == limit_r else None)
      initial = lambda m, r, t: inits[t] if r == limit_r else 0
    else:
      bounds = bounds or (None, None)
      initial = 0
    # production variable depends on resources, time
    #FIXME initials! Should be lambda with mins for tracking var!
    prod = pyo.Var(indexer, self.model.T, initialize=initial, bounds=bounds, **kwargs)
    # TODO it may be that we need to set variable values to avoid problems in some solvers.
    # if comp.is_dispatchable() == 'fixed':
    #   for t, _ in enumerate(m.Times):
    #     prod[limit_r, t].fix(caps[t])
    setattr(self.model, prod_name, prod)
    return prod_name


  def _create_ramp_limit(self, comp, prod_name):
    """
      Creates ramping limitations for a producing component
      @ In, comp, HERON Component, component to make ramping limits for
      @ In, prod_name, str, name of production variable
      @ Out, None
    """
    # ramping is defined in terms of the capacity variable
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    cap = comp._capacity
    r = self.resource_index_map[comp][cap_res] # production index of the governing resource
    # NOTE: this includes the built capacity * capacity factor, if any, which assumes
    # the ramp rate depends on the available capacity, not the built capacity.
    limit_delta = comp.ramp_limit * cap # NOTE: if cap is negative, then this is negative.
    if limit_delta < 0:
      neg_cap = True
    else:
      neg_cap = False
    # if we're limiting ramp frequency, make vars and rules for that
    if comp.ramp_freq:
      # create binaries for tracking ramping
      up = pyo.Var(self.model.T, initialize=0, domain=pyo.Binary)
      down = pyo.Var(self.model.T, initialize=0, domain=pyo.Binary)
      steady = pyo.Var(self.model.T, initialize=1, domain=pyo.Binary)
      setattr(self.model, f'{comp.name}_up_ramp_tracker', up)
      setattr(self.model, f'{comp.name}_down_ramp_tracker', down)
      setattr(self.model, f'{comp.name}_steady_ramp_tracker', steady)
      ramp_trackers = (down, up, steady)
    else:
      ramp_trackers = None
    # limit production changes when ramping down
    ramp_rule_down = lambda mod, t: prl.ramp_rule_down(prod_name, r, limit_delta, neg_cap, t, mod, bins=ramp_trackers)
    constr = pyo.Constraint(self.model.T, rule=ramp_rule_down)
    setattr(self.model, f'{comp.name}_ramp_down_constr', constr)
    # limit production changes when ramping up
    ramp_rule_up = lambda mod, t: prl.ramp_rule_up(prod_name, r, limit_delta, neg_cap, t, mod, bins=ramp_trackers)
    constr = pyo.Constraint(self.model.T, rule=ramp_rule_up)
    setattr(self.model, f'{comp.name}_ramp_up_constr', constr)
    # if ramping frequency limit, impose binary constraints
    if comp.ramp_freq:
      # binaries rule, for exclusive choice up/down/steady
      binaries_rule = lambda mod, t: prl.ramp_freq_bins_rule(down, up, steady, t, mod)
      constr = pyo.Constraint(self.model.T, rule=binaries_rule)
      setattr(self.model, f'{comp.name}_ramp_freq_binaries', constr)
      # limit frequency of ramping
      # TODO calculate "tao" window using ramp freq and dt
      # -> for now, just use the integer for number of windows
      freq_rule = lambda mod, t: prl.ramp_freq_rule(down, up, comp.ramp_freq, t, mod)
      constr = pyo.Constraint(self.model.T, rule=freq_rule)
      setattr(self.model, f'{comp.name}_ramp_freq_constr', constr)


  def _create_capacity_constraints(self, comp, prod_name):
    """
      Creates pyomo capacity constraints
      @ In, comp, HERON Component, component to make variables for
      @ In, prod_name, str, name of production variable
      @ Out, None
    """
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    r = self.resource_index_map[comp][cap_res] # production index of the governing resource
    caps, mins = comp._capacity_vector_t, comp._minimum_vector_t
    # capacity
    max_rule = lambda mod, t: prl.capacity_rule(prod_name, r, caps, mod, t)
    constr = pyo.Constraint(self.model.T, rule=max_rule)
    setattr(self.model, f'{comp.name}_{cap_res}_capacity_constr', constr)
    # minimum
    min_rule = lambda mod, t: prl.min_prod_rule(prod_name, r, caps, mins, mod, t)
    constr = pyo.Constraint(self.model.T, rule=min_rule)
    # set initial conditions
    for t, time in enumerate(self.model.Times):
      cap = caps[t]
      if cap == mins[t]:
        # initialize values so there's no boundary errors
        var = getattr(self.model, prod_name)
        values = var.get_values()
        for k in values:
          values[k] = cap
        var.set_values(values)
    setattr(self.model, f'{comp.name}_{cap_res}_minprod_constr', constr)


  def _create_transfer(self, comp, prod_name):
    """
      Creates pyomo transfer function constraints
      @ In, comp, HERON Component, component to make variables for
      @ In, prod_name, str, name of production variable
      @ Out, None
    """
    transfer = comp.get_interaction().get_transfer()
    if transfer is None:
      return
    if transfer.type == 'Ratio':
      self._create_transfer_ratio(transfer, comp, prod_name)
    elif transfer.type == 'Polynomial':
      self._create_transfer_poly(transfer, comp, prod_name)
    else:
      raise NotImplementedError(f'Transfer function type "{transfer.type}" not implemented for PyomoModelHandler!')

  def _create_transfer_ratio(self, transfer, comp, prod_name):
    """
      Create a balance ratio-based transfer function.
      This comes in the form of a balance expression (not an equality).
      @ In, transfer, TransferFunc, Ratio transfer function
      @ In, comp, Component, component object for this transfer
      @ In, prod_name, str, name of production element
      @ Out, None
    """
    name = comp.name
    coeffs = comp._coeffs
    coeffs_iter = iter(coeffs.items())
    first_name, first_coef = next(coeffs_iter)
    first_r = self.resource_index_map[comp][first_name]
    for resource, coef in coeffs_iter:
      ratio = coef / first_coef
      r = self.resource_index_map[comp][resource]
      rule_name = f'{name}_{resource}_{first_name}_transfer'
      rule = lambda mod, t: prl.ratio_transfer_rule(ratio, r, first_r, prod_name, mod, t)
      constr = pyo.Constraint(self.model.T, rule=rule)
      setattr(self.model, rule_name, constr)

  def _create_transfer_poly(self, transfer, comp, prod_name):
    """
      Create a polynomial transfer function. This comes in the form of an equality expression.
      @ In, transfer, TransferFunc, Ratio transfer function
      @ In, comp, Component, component object for this transfer
      @ In, prod_name, str, name of production element
      @ Out, None
    """
    rule_name = f'{comp.name}_transfer_func'
    coeffs = transfer.get_coefficients()
    # dict of form {(r1, r2): {(o1, o2): n}}
    #   where:
    #   r1, r2 are resource names
    #   o1, o2 are polynomial orders (may not be integers?)
    #   n is the float polynomial coefficient for the term
    rule = lambda mod, t: prl.poly_transfer_rule(coeffs, self.resource_index_map[comp], prod_name, mod, t)
    constr = pyo.Constraint(self.model.T, rule=rule)
    setattr(self.model, rule_name, constr)

  def _create_storage(self, comp):
    """
      Creates storage pyomo variable objects for a storage component
      Similar to create_production, but for storages
      @ In, comp, HERON Component, component to make production variables for
      @ Out, None
    """
    prefix = comp.name
    # what resource index? Isn't it always 0? # assumption
    r = 0 # NOTE this is only true if each storage ONLY uses 1 resource
    # storages require a few variables:
    # (1) a level tracker,
    level_name = self._create_production_variable(comp, tag='level')
    # -> set operational limits
    # self._create_capacity(m, comp, level_name, meta)
    # (2, 3) separate charge/discharge trackers, so we can implement round-trip efficiency and ramp rates
    # Storage charging and/or discharging rates might be limited to less than the component capacity
    max_charge, max_discharge = comp.get_interaction().get_charge_rate_limits(self.meta)
    charge_name = self._create_production_variable(comp,
                                                   tag='charge',
                                                   add_bounds=False,
                                                   bounds=None if not max_charge else (-max_charge, 0),
                                                   within=pyo.NonPositiveReals)
    discharge_name = self._create_production_variable(comp,
                                                      tag='discharge',
                                                      add_bounds=False,
                                                      bounds=None if not max_discharge else (0, max_discharge),
                                                      within=pyo.NonNegativeReals)
    # balance level, charge/discharge
    level_rule_name = prefix + '_level_constr'
    if comp.get_interaction().apply_periodic_level:
      level_var = getattr(self.model, level_name)
      initial = level_var[(r, self.model.T[-1])]
    else:
      initial = self.initial_storage[comp]
    rule = lambda mod, t: prl.level_rule(comp, level_name, charge_name, discharge_name, initial, r, mod, t)
    setattr(self.model, level_rule_name, pyo.Constraint(self.model.T, rule=rule))

    # (4) a binary variable to track whether we're charging or discharging, to prevent BOTH happening
    # -> 0 is charging, 1 is discharging
    # -> TODO make this a user-based option to disable, if they want to allow dual operation
    # -> -> but they should really think about if that's what they want!
    # FIXME currently introducing the bigM strategy also makes solves numerically unstable,
    # and frequently results in spurious errors. For now, disable it.
    allow_both = True # allow simultaneous charging and discharging
    if not allow_both:
      bin_name = self._create_production_variable(comp, tag='dcforcer', add_bounds=False, within=pyo.Binary)
      # we need a large epsilon, but not so large that addition stops making sense
      # -> we don't know what any values for this component will be! How do we choose?
      # -> NOTE that choosing this value has VAST impact on solve stability!!
      large_eps = 1e8 #0.01 * sys.float_info.max
      # charging constraint: don't charge while discharging (note the sign matters)
      charge_rule_name = prefix + '_charge_constr'
      rule = lambda mod, t: prl.charge_rule(charge_name, bin_name, large_eps, r, mod, t)
      setattr(self.model, charge_rule_name, pyo.Constraint(self.model.T, rule=rule))
      discharge_rule_name = prefix + '_discharge_constr'
      rule = lambda mod, t: prl.discharge_rule(discharge_name, bin_name, large_eps, r, mod, t)
      setattr(self.model, discharge_rule_name, pyo.Constraint(self.model.T, rule=rule))


  def _create_conservation(self):
    """
      Creates pyomo conservation constraints
      @ In, None
      @ Out, None
    """
    for resource in self.resources:
      rule = lambda mod, t: prl.conservation_rule(resource, mod, t)
      constr = pyo.Constraint(self.model.T, rule=rule)
      setattr(self.model, f'{resource}_conservation', constr)


  def _create_objective(self):
    """
      Creates pyomo objective function
      @ In, None
      @ Out, None
    """
    # cashflow eval
    rule = lambda mod: prl.cashflow_rule(self._compute_cashflows, self.meta, mod)
    self.model.obj = pyo.Objective(rule=rule, sense=pyo.maximize)

  def _compute_cashflows(self, components, activity, times, meta, state_args=None, time_offset=0):
    """
      Method to compute CashFlow evaluations given components and their activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ In, meta, dict, additional info to be passed through to functional evaluations
      @ In, state_args, dict, optional, additional arguments to pass while getting activity state
      @ In, time_offset, int, optional, increase time index tracker by this value if provided
      @ Out, total, float, total cashflows for given components
    """
    if state_args is None:
      state_args = {}

    # dispatch to levelized if requested
    if self.case.use_levelized_inner:
      return self._compute_levelized_cashflows(
        components, activity, times, meta, state_args, time_offset
      )

    total = 0.0

    # only consider components that have cashflows
    comps_with_recurring_cfs = []
    for comp in components:
      for cf in comp.get_cashflows():
        if cf.get_type() == 'repeating' and cf.get_period() != "year":
          comps_with_recurring_cfs.append(comp)
          continue

    for comp in comps_with_recurring_cfs:
      cfs = []
      for t, time in enumerate(times):
        for tracker in comp.get_tracking_vars():
          for res in self.resource_index_map[comp]:
            pyo_activity = self._build_specific_activity(comp, activity, time, state_args)
            cfs.append(comp._alpha_vector_t[t] * (pyo_activity[tracker][res] / comp._dprime_vector_t[t])**comp._scaling_factor_vector_t[t])
        total += sum(cfs)

    return total


  def _build_specific_activity(self, comp, activity, time, state_args):
    """
      Helper to build the activity dict for a component at a given time.
    """
    snapshot = {}
    for tracker in comp.get_tracking_vars():
      snapshot[tracker] = {
        resource: activity.get_activity(
          comp, tracker, resource, time, **state_args
        )
        for resource in self.resource_index_map[comp]
      }
    return snapshot

  def _compute_levelized_cashflows(self, components, activity, times, meta, state_args=None, time_offset=0):
    """
      Compute levelized cashflows by solving non_multiplied + x * multiplied = npv_target.
      Returns x (with a sign flip to match the cashflow_rule convention).
    """
    total_non = 0.0
    total_mul = 0.0

    # only consider components that have cashflows
    comps_with_recurring_cfs = []
    for comp in components:
      for cf in comp.get_cashflows():
        if cf.get_type() == 'repeating' and cf.get_period() != "year":
          comps_with_recurring_cfs.append(comp)
          comp._is_levelized = cf.is_mult_target()
          continue

    for comp in comps_with_recurring_cfs:
      comp_non = 0.0
      comp_mul = 0.0
      cfs = []
      for t, time in enumerate(times):
        for tracker in comp.get_tracking_vars():
          for res in self.resource_index_map[comp]:
            pyo_activity = self._build_specific_activity(comp, activity, time, state_args)
            cfs.append(comp._alpha_vector_t[t] * (pyo_activity[tracker][res] / comp._dprime_vector_t[t])**comp._scaling_factor_vector_t[t])

        if comp.levelized_meta:
          # extract the levelized cashflow term(s)
          for lvl_key in comp.levelized_meta:
            comp_mul += cfs.pop(lvl_key, 0.0)
        else:
          comp_non += sum(cfs)

      total_non += comp_non
      total_mul += comp_mul

    target = self.case.npv_target
    # solve: total_non + x * total_mul = target  =>  x = (target - total_non) / (total_mul + eps)
    return -(target - total_non) / (total_mul + self._eps)

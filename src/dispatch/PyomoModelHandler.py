# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  This module constructs the dispatch optimization model used by HERON.
"""
import numpy as np
import pyomo.environ as pyo

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
    for comp in self.components:
      self._process_component(comp)
    self._create_conservation() # conservation of resources (e.g. production == consumption)
    self._create_objective() # objective function


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


  def _create_production_variable(self, comp, tag=None, add_bounds=True, **kwargs):
    """
      Creates production pyomo variable object for a component
      @ In, comp, HERON Component, component to make production variables for
      @ In, tag, str, optional, if not None then name will be component_[tag]; otherwise "production"
      @ In, add_bounds, bool, optional, if True then determine and set bounds for variable
      @ In, kwargs, dict, optional, passalong kwargs to pyomo variable
      @ Out, prod_name, str, name of production variable
    """
    if tag is None:
      tag = 'production'
    name = comp.name
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    limit_r = self.model.resource_index_map[comp][cap_res] # production index of the governing resource
    # create pyomo indexer for this component's resources
    indexer_name = f'{name}_res_index_map'
    indexer = getattr(self.model, indexer_name, None)
    if indexer is None:
      indexer = pyo.Set(initialize=range(len(self.model.resource_index_map[comp])))
      setattr(self.model, indexer_name, indexer)
    prod_name = f'{name}_{tag}'
    caps, mins = self._find_production_limits(comp)
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
      bounds = (None, None)
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
    cap = comp.get_capacity(self.meta)[0][cap_res]
    r = self.model.resource_index_map[comp][cap_res] # production index of the governing resource
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
    r = self.model.resource_index_map[comp][cap_res] # production index of the governing resource
    caps, mins = self._find_production_limits(comp)
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


  def _find_production_limits(self, comp):
    """
      Determines the capacity limits of a unit's operation, in time.
      @ In, comp, HERON Component, component to make variables for
      @ Out, caps, array, max production values by time
      @ Out, mins, array, min production values by time
    """
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    r = self.model.resource_index_map[comp][cap_res] # production index of the governing resource
    # production is always lower than capacity
    ## NOTE get_capacity returns (data, meta) and data is dict
    ## TODO does this work with, e.g., ARMA-based capacities?
    ### -> "time" is stored on "m" and could be used to correctly evaluate the capacity
    caps = []
    mins = []
    for t, time in enumerate(self.model.Times):
      self.meta['HERON']['time_index'] = t + self.model.time_offset
      cap = comp.get_capacity(self.meta)[0][cap_res] # value of capacity limit (units of governing resource)
      caps.append(cap)
      if (comp.is_dispatchable() == 'fixed'):
        minimum = cap
      else:
        minimum = comp.get_minimum(self.meta)[0][cap_res]
      mins.append(minimum)
    return caps, mins


  def _create_transfer(self, comp, prod_name):
    """
      Creates pyomo transfer function constraints
      @ In, comp, HERON Component, component to make variables for
      @ In, prod_name, str, name of production variable
      @ Out, None
    """
    name = comp.name
    # transfer functions
    # e.g. 2A + 3B -> 1C + 2E
    # get linear coefficients
    # TODO this could also take a transfer function from an external Python function assuming
    #    we're careful about how the expression-vs-float gets used
    #    and figure out how to handle multiple ins, multiple outs
    ratios = putils.get_transfer_coeffs(self.model, comp)
    ref_r, ref_name, _ = ratios.pop('__reference', (None, None, None))
    for resource, ratio in ratios.items():
      r = self.model.resource_index_map[comp][resource]
      rule_name = f'{name}_{resource}_{ref_name}_transfer'
      rule = lambda mod, t: prl.transfer_rule(ratio, r, ref_r, prod_name, mod, t)
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
    charge_name = self._create_production_variable(comp, tag='charge', add_bounds=False, within=pyo.NonPositiveReals)
    discharge_name = self._create_production_variable(comp, tag='discharge', add_bounds=False, within=pyo.NonNegativeReals)
    # balance level, charge/discharge
    level_rule_name = prefix + '_level_constr'
    rule = lambda mod, t: prl.level_rule(comp, level_name, charge_name, discharge_name, self.initial_storage, r, mod, t)
    setattr(self.model, level_rule_name, pyo.Constraint(self.model.T, rule=rule))
    # periodic boundary condition for storage level
    if comp.get_interaction().apply_periodic_level:
      periodic_rule_name = prefix + '_level_periodic_constr'
      rule = lambda mod, t: prl.periodic_level_rule(comp, level_name, self.initial_storage, r, mod, t)
      setattr(self.model, periodic_rule_name, pyo.Constraint(self.model.T, rule=rule))

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

    if meta['HERON']['Case'].use_levelized_inner:
      total = self._compute_levelized_cashflows(components, activity, times, meta, state_args, time_offset)
      return total

    total = 0
    specific_meta = dict(meta) # TODO what level of copying do we need here?
    resource_indexer = meta['HERON']['resource_indexer']

    #print('DEBUGG computing cashflows!')
    for comp in components:
      #print(f'DEBUGG ... comp {comp.name}')
      specific_meta['HERON']['component'] = comp
      comp_subtotal = 0
      for t, time in enumerate(times):
        #print(f'DEBUGG ... ... time {t}')
        # NOTE care here to assure that pyomo-indexed variables work here too
        specific_activity = {}
        for tracker in comp.get_tracking_vars():
          specific_activity[tracker] = {}
          for resource in resource_indexer[comp]:
            specific_activity[tracker][resource] = activity.get_activity(comp, tracker, resource, time, **state_args)
        specific_meta['HERON']['time_index'] = t + time_offset
        specific_meta['HERON']['time_value'] = time
        cfs = comp.get_state_cost(specific_activity, specific_meta, marginal=True)
        time_subtotal = sum(cfs.values())
        comp_subtotal += time_subtotal
      total += comp_subtotal
    return total

  def _compute_levelized_cashflows(self, components, activity, times, meta, state_args=None, time_offset=0):
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
    total = 0
    specific_meta = dict(meta) # TODO what level of copying do we need here?
    resource_indexer = meta['HERON']['resource_indexer']

    # How does this work?
    #   The general equation looks like:
    #
    #     SUM(Non-Multiplied Terms) + x * SUM(Multiplied Terms) = Target
    #
    #   and we are solving for `x`. Target is 0 by default. Terms here are marginal cashflows.
    #   Summations here occur over: components, time steps, tracking variables, and resources.
    #   Typically, there is only 1 multiplied term/cash flow.

    multiplied = 0
    non_multiplied = 0

    for comp in components:
      specific_meta['HERON']['component'] = comp
      multiplied_comp = 0
      non_multiplied_comp = 0
      for t, time in enumerate(times):
        # NOTE care here to assure that pyomo-indexed variables work here too
        specific_activity = {}
        for tracker in comp.get_tracking_vars():
          specific_activity[tracker] = {}
          for resource in resource_indexer[comp]:
            specific_activity[tracker][resource] = activity.get_activity(comp, tracker, resource, time, **state_args)
        specific_meta['HERON']['time_index'] = t + time_offset
        specific_meta['HERON']['time_value'] = time
        cfs = comp.get_state_cost(specific_activity, specific_meta, marginal=True)

        # there is an assumption here that if a component has a levelized cost, marginal cashflow
        # then it is the only marginal cashflow
        if comp.levelized_meta:
          for cf in comp.levelized_meta.keys():
            lcf = cfs.pop(cf) # this should be ok as long as HERON init checks are successful
            multiplied_comp += lcf
        else:
          time_subtotal = sum(cfs.values())
          non_multiplied_comp += time_subtotal

      multiplied     += multiplied_comp
      non_multiplied += non_multiplied_comp

    # at this point, there should be a not None NPV Target
    multiplied += self._eps
    total = (meta['HERON']['Case'].npv_target - non_multiplied) / multiplied
    total *= -1
    return total


import numpy as np
import pyomo.environ as pyo

from . import PyomoRuleLibrary as prl
from . import putils
from .DispatchState import PyomoState

class PyomoModelHandler:

  def __init__(self, time, time_offset, case, components, resources, initial_storage, meta) -> None:
    self.time = time
    self.time_offset = time_offset
    self.case = case
    self.components = components
    # self.sources = sources
    self.resources = resources
    self.initial_storage = initial_storage
    self.meta = meta
    self.model = pyo.ConcreteModel()

  def build_model(self):
    C = np.arange(0, len(self.components), dtype=int) # indexes component
    R = np.arange(0, len(self.resources), dtype=int) # indexes resources
    T = np.arange(0, len(self.time), dtype=int) # indexes resources
    self.model.C = pyo.Set(initialize=C)
    self.model.R = pyo.Set(initialize=R)
    self.model.T = pyo.Set(initialize=T)
    self.model.Times = self.time
    self.model.time_offset = self.time_offset
    self.model.resource_index_map = self.meta['HERON']['resource_indexer'] # maps the resource to its index WITHIN APPLICABLE components (sparse matrix) e.g. component: {resource: local index}, ... etc}
    # properties
    self.model.Case = self.case
    self.model.Components = self.components
    self.model.Activity = PyomoState()
    self.model.Activity.initialize(self.model.Components, self.model.resource_index_map, self.model.Times, self.model)

  def populate_model(self):
    for comp in self.components:
      self._process_component(comp)
    self._create_conservation() # conservation of resources (e.g. production == consumption)
    self._create_objective() # objective function

  def _process_component(self, component):
    interaction = component.get_interaction()
    if interaction.is_governed():
      self._process_governed_component(component, interaction)
    elif interaction.is_type("Storage"):
      self._create_storage(component)
    else:
      self._create_production(component)

  def _process_governed_component(self, component, interaction):
    """
    """
    self.meta["request"] = {"component": component, "time": self.time}
    if interaction.is_type("Storage"):
      self._process_storage_component(component, interaction)
    else:
      activity = interaction.get_strategy().evaluate(self.meta)[0]['level']
      self._create_production_param(component, activity)

  def _process_storage_component(self, m, component, interaction):
    activity = interaction.get_strategy().evaluate(self.meta)[0]["level"]
    self._create_production_param(m, component, activity, tag="level")
    dt = self.model.Times[1] - self.model.Times[0]
    rte2 = component.get_sqrt_RTE()
    deltas = self._calculate_deltas(activity, interaction.get_initial_level(self.meta))
    charge = np.where(deltas > 0, -deltas / dt / rte2, 0)
    discharge = np.where(deltas < 0, -deltas / dt * rte2, 0)
    self._create_production_param(m, component, charge, tag="charge")
    self._create_production_param(m, component, discharge, tag="discharge")

 ### PYOMO Element Constructors
  def _create_production_limit(self, m, validation):
    """
      Creates pyomo production constraint given validation errors
      @ In, m, pyo.ConcreteModel, associated model
      @ In, validation, dict, information from Validator about limit violation
      @ Out, None
    """
    # TODO could validator write a symbolic expression on request? That'd be sweet.
    comp = validation['component']
    resource = validation['resource']
    r = m.resource_index_map[comp][resource]
    t = validation['time_index']
    limit = validation['limit']
    limits = np.zeros(len(m.Times))
    limits[t] = limit
    limit_type = validation['limit_type']
    prod_name = f'{comp.name}_production'
    rule = lambda mod: prl.prod_limit_rule(prod_name, r, limits, limit_type, t, mod)
    constr = pyo.Constraint(rule=rule)
    counter = 1
    name_template = f'{comp.name}_{resource}_{t}_vld_limit_constr_{{i}}'
    # make sure we get a unique name for this constraint
    name = name_template.format(i=counter)
    while getattr(m, name, None) is not None:
      counter += 1
      name = name_template.format(i=counter)
    setattr(m, name, constr)
    print(f'DEBUGG added validation constraint "{name}"')


  def _create_production_param(self, m, comp, values, tag=None):
    """
      Creates production pyomo fixed parameter object for a component
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make production variables for
      @ In, values, np.array(float), values to set for param
      @ In, tag, str, optional, if not None then name will be component_[tag]
      @ Out, prod_name, str, name of production variable
    """
    name = comp.name
    if tag is None:
      tag = 'production'
    # create pyomo indexer for this component's resources
    res_indexer = pyo.Set(initialize=range(len(m.resource_index_map[comp])))
    setattr(m, f'{name}_res_index_map', res_indexer)
    prod_name = f'{name}_{tag}'
    init = (((0, t), values[t]) for t in m.T)
    prod = pyo.Param(res_indexer, m.T, initialize=dict(init))
    setattr(m, prod_name, prod)
    return prod_name


  def _create_production(self, m, comp, meta):
    """
      Creates all pyomo variable objects for a non-storage component
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make production variables for
      @ In, meta, dict, dictionary of state variables
      @ Out, None
    """
    prod_name = self._create_production_variable(m, comp, meta)
    ## if you cannot set limits directly in the production variable, set separate contraint:
    ## Method 1: set variable bounds directly --> TODO more work needed, but would be nice
    # lower, upper = self._get_prod_bounds(m, comp)
    # limits should be None unless specified, so use "getters" from dictionaries
    # bounds = lambda m, r, t: (lower.get(r, None), upper.get(r, None))
    ## Method 2: set variable bounds directly --> TODO more work needed, but would be nice
    # self._create_capacity(m, comp, prod_name, meta)    # capacity constraints
    # transfer function governs input -> output relationship
    self._create_transfer(m, comp, prod_name)
    # ramp rates
    if comp.ramp_limit is not None:
      self._create_ramp_limit(m, comp, prod_name, meta)
    return prod_name


  def _create_production_variable(self, m, comp, meta, tag=None, add_bounds=True, **kwargs):
    """
      Creates production pyomo variable object for a component
      @ In, m, pyo.ConcreteModel, associated model
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
    limit_r = m.resource_index_map[comp][cap_res] # production index of the governing resource
    # create pyomo indexer for this component's resources
    indexer_name = f'{name}_res_index_map'
    indexer = getattr(m, indexer_name, None)
    if indexer is None:
      indexer = pyo.Set(initialize=range(len(m.resource_index_map[comp])))
      setattr(m, indexer_name, indexer)
    prod_name = f'{name}_{tag}'
    caps, mins = self._find_production_limits(m, comp, meta)
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
    prod = pyo.Var(indexer, m.T, initialize=initial, bounds=bounds, **kwargs)
    # TODO it may be that we need to set variable values to avoid problems in some solvers.
    # if comp.is_dispatchable() == 'fixed':
    #   for t, _ in enumerate(m.Times):
    #     prod[limit_r, t].fix(caps[t])
    setattr(m, prod_name, prod)
    return prod_name


  def _create_ramp_limit(self, m, comp, prod_name, meta):
    """
      Creates ramping limitations for a producing component
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make ramping limits for
      @ In, prod_name, str, name of production variable
      @ In, meta, dict, dictionary of state variables
      @ Out, None
    """
    # ramping is defined in terms of the capacity variable
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    cap = comp.get_capacity(meta)[0][cap_res]
    r = m.resource_index_map[comp][cap_res] # production index of the governing resource
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
      up = pyo.Var(m.T, initialize=0, domain=pyo.Binary)
      down = pyo.Var(m.T, initialize=0, domain=pyo.Binary)
      steady = pyo.Var(m.T, initialize=1, domain=pyo.Binary)
      setattr(m, f'{comp.name}_up_ramp_tracker', up)
      setattr(m, f'{comp.name}_down_ramp_tracker', down)
      setattr(m, f'{comp.name}_steady_ramp_tracker', steady)
      ramp_trackers = (down, up, steady)
    else:
      ramp_trackers = None
    # limit production changes when ramping down
    ramp_rule_down = lambda mod, t: prl.ramp_rule_down(prod_name, r, limit_delta, neg_cap, t, mod, bins=ramp_trackers)
    constr = pyo.Constraint(m.T, rule=ramp_rule_down)
    setattr(m, f'{comp.name}_ramp_down_constr', constr)
    # limit production changes when ramping up
    ramp_rule_up = lambda mod, t: prl.ramp_rule_up(prod_name, r, limit_delta, neg_cap, t, mod, bins=ramp_trackers)
    constr = pyo.Constraint(m.T, rule=ramp_rule_up)
    setattr(m, f'{comp.name}_ramp_up_constr', constr)
    # if ramping frequency limit, impose binary constraints
    if comp.ramp_freq:
      # binaries rule, for exclusive choice up/down/steady
      binaries_rule = lambda mod, t: prl.ramp_freq_bins_rule(down, up, steady, t, mod)
      constr = pyo.Constraint(m.T, rule=binaries_rule)
      setattr(m, f'{comp.name}_ramp_freq_binaries', constr)
      # limit frequency of ramping
      # TODO calculate "tao" window using ramp freq and dt
      # -> for now, just use the integer for number of windows
      freq_rule = lambda mod, t: prl.ramp_freq_rule(down, up, comp.ramp_freq, t, m)
      constr = pyo.Constraint(m.T, rule=freq_rule)
      setattr(m, f'{comp.name}_ramp_freq_constr', constr)


  def _create_capacity_constraints(self, m, comp, prod_name, meta):
    """
      Creates pyomo capacity constraints
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make variables for
      @ In, prod_name, str, name of production variable
      @ In, meta, dict, additional state information
      @ Out, None
    """
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    r = m.resource_index_map[comp][cap_res] # production index of the governing resource
    caps, mins = self._find_production_limits(m, comp, meta)
    # capacity
    max_rule = lambda mod, t: prl.capacity_rule(prod_name, r, caps, mod, t)
    constr = pyo.Constraint(m.T, rule=max_rule)
    setattr(m, f'{comp.name}_{cap_res}_capacity_constr', constr)
    # minimum
    min_rule = lambda mod, t: prl.min_prod_rule(prod_name, r, caps, mins, mod, t)
    constr = pyo.Constraint(m.T, rule=min_rule)
    # set initial conditions
    for t, time in enumerate(m.Times):
      cap = caps[t]
      if cap == mins[t]:
        # initialize values so there's no boundary errors
        var = getattr(m, prod_name)
        values = var.get_values()
        for k in values:
          values[k] = cap
        var.set_values(values)
    setattr(m, f'{comp.name}_{cap_res}_minprod_constr', constr)


  def _find_production_limits(self, m, comp, meta):
    """
      Determines the capacity limits of a unit's operation, in time.
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make variables for
      @ In, meta, dict, additional state information
      @ Out, caps, array, max production values by time
      @ Out, mins, array, min production values by time
    """
    cap_res = comp.get_capacity_var()       # name of resource that defines capacity
    r = m.resource_index_map[comp][cap_res] # production index of the governing resource
    # production is always lower than capacity
    ## NOTE get_capacity returns (data, meta) and data is dict
    ## TODO does this work with, e.g., ARMA-based capacities?
    ### -> "time" is stored on "m" and could be used to correctly evaluate the capacity
    caps = []
    mins = []
    for t, time in enumerate(m.Times):
      meta['HERON']['time_index'] = t + m.time_offset
      cap = comp.get_capacity(meta)[0][cap_res] # value of capacity limit (units of governing resource)
      caps.append(cap)
      if (comp.is_dispatchable() == 'fixed'):
        minimum = cap
      else:
        minimum = comp.get_minimum(meta)[0][cap_res]
      mins.append(minimum)
    return caps, mins


  def _create_transfer(self, m, comp, prod_name):
    """
      Creates pyomo transfer function constraints
      @ In, m, pyo.ConcreteModel, associated model
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
    ratios = putils.get_transfer_coeffs(m, comp)
    ref_r, ref_name, _ = ratios.pop('__reference', (None, None, None))
    for resource, ratio in ratios.items():
      r = m.resource_index_map[comp][resource]
      rule_name = f'{name}_{resource}_{ref_name}_transfer'
      rule = lambda mod, t: prl.transfer_rule(ratio, r, ref_r, prod_name, mod, t)
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, rule_name, constr)


  def _create_storage(self, m, comp, initial_storage, meta):
    """
      Creates storage pyomo variable objects for a storage component
      Similar to create_production, but for storages
      @ In, m, pyo.ConcreteModel, associated model
      @ In, comp, HERON Component, component to make production variables for
      @ In, initial_storage, dict, initial storage levels
      @ In, meta, dict, additional state information
      @ Out, level_name, str, name of storage level variable
    """
    prefix = comp.name
    # what resource index? Isn't it always 0? # assumption
    r = 0 # NOTE this is only true if each storage ONLY uses 1 resource
    # storages require a few variables:
    # (1) a level tracker,
    level_name = self._create_production_variable(m, comp, meta, tag='level')
    # -> set operational limits
    # self._create_capacity(m, comp, level_name, meta)
    # (2, 3) separate charge/discharge trackers, so we can implement round-trip efficiency and ramp rates
    charge_name = self._create_production_variable(m, comp, meta, tag='charge', add_bounds=False, within=pyo.NonPositiveReals)
    discharge_name = self._create_production_variable(m, comp, meta, tag='discharge', add_bounds=False, within=pyo.NonNegativeReals)
    # balance level, charge/discharge
    level_rule_name = prefix + '_level_constr'
    rule = lambda mod, t: prl.level_rule(comp, level_name, charge_name, discharge_name, initial_storage, r, mod, t)
    setattr(m, level_rule_name, pyo.Constraint(m.T, rule=rule))
    # periodic boundary condition for storage level
    if comp.get_interaction().apply_periodic_level:
      periodic_rule_name = prefix + '_level_periodic_constr'
      rule = lambda mod, t: prl.periodic_level_rule(comp, level_name, initial_storage, r, mod, t)
      setattr(m, periodic_rule_name, pyo.Constraint(m.T, rule=rule))

    # (4) a binary variable to track whether we're charging or discharging, to prevent BOTH happening
    # -> 0 is charging, 1 is discharging
    # -> TODO make this a user-based option to disable, if they want to allow dual operation
    # -> -> but they should really think about if that's what they want!
    # FIXME currently introducing the bigM strategy also makes solves numerically unstable,
    # and frequently results in spurious errors. For now, disable it.
    allow_both = True # allow simultaneous charging and discharging
    if not allow_both:
      bin_name = self._create_production_variable(m, comp, meta, tag='dcforcer', add_bounds=False, within=pyo.Binary)
      # we need a large epsilon, but not so large that addition stops making sense
      # -> we don't know what any values for this component will be! How do we choose?
      # -> NOTE that choosing this value has VAST impact on solve stability!!
      large_eps = 1e8 #0.01 * sys.float_info.max
      # charging constraint: don't charge while discharging (note the sign matters)
      charge_rule_name = prefix + '_charge_constr'
      rule = lambda mod, t: prl.charge_rule(charge_name, bin_name, large_eps, r, mod, t)
      setattr(m, charge_rule_name, pyo.Constraint(m.T, rule=rule))
      discharge_rule_name = prefix + '_discharge_constr'
      rule = lambda mod, t: prl.discharge_rule(discharge_name, bin_name, large_eps, r, mod, t)
      setattr(m, discharge_rule_name, pyo.Constraint(m.T, rule=rule))


  def _create_conservation(self, m, resources):
    """
      Creates pyomo conservation constraints
      @ In, m, pyo.ConcreteModel, associated model
      @ In, resources, list, list of resources in problem
      @ In, initial_storage, dict, initial storage levels
      @ In, meta, dict, dictionary of state variables
      @ Out, None
    """
    for resource in resources:
      rule = lambda mod, t: prl.conservation_rule(resource, mod, t)
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, f'{resource}_conservation', constr)


  def _create_objective(self, meta, m):
    """
      Creates pyomo objective function
      @ In, meta, dict, additional variables to pass through
      @ In, m, pyo.ConcreteModel, associated model
      @ Out, None
    """
    # cashflow eval
    rule = lambda mod: prl.cashflow_rule(self._compute_cashflows, meta, mod)
    m.obj = pyo.Objective(rule=rule, sense=pyo.maximize)

  
  

    
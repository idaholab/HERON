# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Library of Pyomo rules for HERON dispatch.
"""
import pyomo.environ as pyo

def charge_rule(charge_name, bin_name, large_eps, r, m, t) -> bool:
  """
    Constructs pyomo don't-charge-while-discharging constraints.
    For storage units specificially.
    @ In, charge_name, str, name of charging variable
    @ In, bin_name, str, name of forcing binary variable
    @ In, large_eps, float, a large-ish number w.r.t. storage activity
    @ In, r, int, index of stored resource (is this always 0?)
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, time index for capacity rule
    @ Out, rule, bool, inequality used to limit charge behavior
  """
  charge_var = getattr(m, charge_name)
  bin_var = getattr(m, bin_name)
  return -charge_var[r, t] <= (1 - bin_var[r, t]) * large_eps

def discharge_rule(discharge_name, bin_name, large_eps, r, m, t) -> bool:
  """
    Constructs pyomo don't-discharge-while-charging constraints.
    For storage units specificially.
    @ In, discharge_name, str, name of discharging variable
    @ In, bin_name, str, name of forcing binary variable
    @ In, large_eps, float, a large-ish number w.r.t. storage activity
    @ In, r, int, index of stored resource (is this always 0?)
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, time index for capacity rule
    @ Out, rule, bool, inequality used to limit discharge behavior
  """
  discharge_var = getattr(m, discharge_name)
  bin_var = getattr(m, bin_name)
  return discharge_var[r, t] <= bin_var[r, t] * large_eps

def level_rule(comp, level_name, charge_name, discharge_name, initial_storage, r, m, t) -> bool:
  """
    Constructs pyomo charge-discharge-level balance constraints.
    For storage units specificially.
    @ In, comp, Component, storage component of interest
    @ In, level_name, str, name of level-tracking variable
    @ In, charge_name, str, name of charging variable
    @ In, discharge_name, str, name of discharging variable
    @ In, initial_storage, dict, initial storage levels by component
    @ In, r, int, index of stored resource (is this always 0?)
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, time index for capacity rule
    @ Out, rule, bool, inequality used to limit level behavior
  """
  level_var = getattr(m, level_name)
  charge_var = getattr(m, charge_name)
  discharge_var = getattr(m, discharge_name)
  if t > 0:
    previous = level_var[r, t-1]
    dt = m.Times[t] - m.Times[t-1]
  else:
    previous = initial_storage[comp]
    dt = m.Times[1] - m.Times[0]
  rte2 = comp.get_sqrt_RTE() # square root of the round-trip efficiency
  production = - rte2 * charge_var[r, t] - discharge_var[r, t] / rte2
  return level_var[r, t] == previous + production * dt

def periodic_level_rule(comp, level_name, initial_storage, r, m, t) -> bool:
  """
    Mandates storage units end with the same level they start with, which prevents
    "free energy" or "free sink" due to initial starting levels.
    For storage units specificially.
    @ In, comp, Component, storage component of interest
    @ In, level_name, str, name of level-tracking variable
    @ In, initial_storage, dict, initial storage levels by component
    @ In, r, int, index of stored resource (is this always 0?)
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, time index for capacity rule
    @ Out, rule, bool, inequality used to limit level behavior
  """
  return getattr(m, level_name)[r, m.T[-1]] == initial_storage[comp]

def capacity_rule(prod_name, r, caps, m, t) -> bool:
  """
    Constructs pyomo capacity constraints.
    @ In, prod_name, str, name of production variable
    @ In, r, int, index of resource for capacity constraining
    @ In, caps, list(float), value to constrain resource at in time
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, time index for capacity rule
  """
  kind = 'lower' if min(caps) < 0 else 'upper'
  return prod_limit_rule(prod_name, r, caps, kind, t, m)

def prod_limit_rule(prod_name, r, limits, kind, t, m) -> bool:
  """
    Constructs pyomo production constraints.
    @ In, prod_name, str, name of production variable
    @ In, r, int, index of resource for capacity constraining
    @ In, limits, list(float), values in time at which to constrain resource production
    @ In, kind, str, either 'upper' or 'lower' for limiting production
    @ In, t, int, time index for production rule (NOTE not pyomo index, rather fixed index)
    @ In, m, pyo.ConcreteModel, associated model
    @ Out, rule, bool, pyomo expression contraint for production limits
  """
  prod = getattr(m, prod_name)
  if kind == 'lower':
    # production must exceed value
    return prod[r, t] >= limits[t]
  elif kind == 'upper':
    return prod[r, t] <= limits[t]
  else:
    raise TypeError('Unrecognized production limit "kind":', kind)

def ramp_rule_down(prod_name, r, limit, neg_cap, t, m, bins=None) -> bool:
  """
    Constructs pyomo production ramping constraints for reducing production level.
    Note that this is number-getting-less-positive for positive-defined capacity, while
      it is number-getting-less-negative for negative-defined capacity.
    This means that dQ is negative for positive-defined capacity, but positive for vice versa
    @ In, prod_name, str, name of production variable
    @ In, r, int, index of resource for capacity constraining
    @ In, limit, float, limiting change in production level across time steps. NOTE: negative for negative-defined capacity.
    @ In, neg_cap, bool, True if capacity is expressed as negative (consumer)
    @ In, t, int, time index for ramp limit rule (NOTE not pyomo index, rather fixed index)
    @ In, m, pyo.ConcreteModel, associated model
    @ In, bins, tuple, optional, (lower, upper, steady) binaries if limiting ramp frequency
    @ Out, rule, expression, evaluation for Pyomo constraint
  """
  prod = getattr(m, prod_name)
  if t == 0:
    return pyo.Constraint.Skip
  delta = prod[r, t] - prod[r, t-1]
  # special treatment if we have frequency-limiting binaries available
  if bins is None:
    if neg_cap:
      # NOTE change in production should be "less positive" than the max
      #   "negative decrease" in production (decrease is positive when defined by consuming)
      return delta <= - limit
    else:
      # dq is negative, - limit is negative
      return delta >= - limit
  else:
    eps = 1.0 # aux parameter to force binaries to behave, TODO needed?
    down = bins[0][t]
    up = bins[1][t]
    # NOTE we're following the convention that "less negative" is ramping "down"
    #   for capacity defined by consumption
    #   e.g. consuming 100 ramps down to consuming 70 is (-100 -> -70), dq = 30
    if neg_cap:
      # dq <= limit * dt * Bu + eps * Bd, if limit <= 0
      # dq is positive, - limit is positive
      return delta <= - limit * down - eps * up
    else:
      # dq <= limit * dt * Bu - eps * Bd, if limit >= 0
      # dq is negative, - limit is negative
      return delta >= - limit * down + eps * up

def ramp_rule_up(prod_name, r, limit, neg_cap, t, m, bins=None) -> bool:
  """
    Constructs pyomo production ramping constraints.
    @ In, prod_name, str, name of production variable
    @ In, r, int, index of resource for capacity constraining
    @ In, limit, float, limiting change in production level across time steps
    @ In, neg_cap, bool, True if capacity is expressed as negative (consumer)
    @ In, t, int, time index for ramp limit rule (NOTE not pyomo index, rather fixed index)
    @ In, m, pyo.ConcreteModel, associated model
    @ In, bins, tuple, optional, (lower, steady, upper) binaries if limiting ramp frequency
    @ Out, rule, expression, evaluation for Pyomo constraint
  """
  prod = getattr(m, prod_name)
  if t == 0:
    return pyo.Constraint.Skip
  delta = prod[r, t] - prod[r, t-1]
  if bins is None:
    if neg_cap:
      # NOTE change in production should be "more positive" than the max
      #   "negative increase" in production (increase is negative when defined by consuming)
      return delta >= limit
    else:
      # change in production should be less than the max production increase
      return delta <= limit
  else:
    # special treatment if we have frequency-limiting binaries available
    eps = 1.0 # aux parameter to force binaries to behave, TODO needed?
    down = bins[0][t]
    up = bins[1][t]
    # NOTE we're following the convention that "more negative" is ramping "up"
    #   for capacity defined by consumption
    #   e.g. consuming 100 ramps up to consuming 130 is (-100 -> -130), dq = -30
    if neg_cap:
      # dq >= limit * dt * Bu + eps * Bd, if limit <= 0
      return delta >= limit * up + eps * down
    else:
      # dq <= limit * dt * Bu - eps * Bd, if limit >= 0
      return delta <= limit * up - eps * down

def ramp_freq_rule(Bd, Bu, tao, t, m) -> bool:
  """
    Constructs pyomo frequency-of-ramp constraints.
    @ In, Bd, bool var, binary tracking down-ramp events
    @ In, Bu, bool var, binary tracking up-ramp events
    @ In, tao, int, number of time steps to look back
    @ In, t, int, time step indexer
    @ In, m, pyo.ConcreteModel, pyomo model
    @ Out, rule, expression, evaluation for Pyomo constraint
  """
  if t == 0:
    return pyo.Constraint.Skip
  # looking-back-window shouldn't be longer than existing time
  tao = min(t, tao)
  # how many ramp-down events in backward window?
  tally = sum(1 - Bd[tm] for tm in range(t - tao, t))
  # only allow ramping up if no rampdowns in back window
  ## but we can't use if statements, so use binary math
  return Bu[t] <= 1 / tao * tally

def ramp_freq_bins_rule(Bd, Bu, Bn, t, m) -> bool:
  """
    Constructs pyomo constraint for ramping event tracking variables.
    This forces choosing between ramping up, ramping down, and steady state operation.
    @ In, Bd, bool var, binary tracking down-ramp events
    @ In, Bu, bool var, binary tracking up-ramp events
    @ In, Bn, bool var, binary tracking no-ramp events
    @ In, t, int, time step indexer
    @ In, m, pyo.ConcreteModel, pyomo model
    @ Out, rule, expression, evaluation for Pyomo constraint
  """
  return Bd[t] + Bu[t] + Bn[t] == 1

def cashflow_rule(compute_cashflows, meta, m) -> float:
  """
    Objective function rule.
    @ In, compute_cashflows, function, function to compute cashflows
    @ In, meta, dict, additional variable passthrough
    @ In, m, pyo.ConcreteModel, associated model
    @ Out, total, float, evaluation of cost
  """
  activity = m.Activity
  state_args = {'valued': False}
  total = compute_cashflows(m.Components, activity, m.Times, meta, state_args=state_args, time_offset=m.time_offset)
  return total

def conservation_rule(res, m, t) -> bool:
  """
    Constructs conservation constraints.
    @ In, res, str, name of resource
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, index of time variable
    @ Out, conservation, bool, balance check
  """
  balance = 0 # sum of production rates, which needs to be zero
  for comp, res_dict in m.resource_index_map.items():
    if res in res_dict:
      # activity information depends on if storage or component
      r = res_dict[res]
      intr = comp.get_interaction()
      if intr.is_type('Storage'):
        # Storages have 3 variables: level, charge, and discharge
        # -> so calculate activity
        charge = getattr(m, f'{comp.name}_charge')
        discharge = getattr(m, f'{comp.name}_discharge')
        # note that "charge" is negative (as it's consuming) and discharge is positive
        # -> so the intuitive |discharge| - |charge| becomes discharge + charge
        production = discharge[r, t] + charge[r, t]
      else:
        var = getattr(m, f'{comp.name}_production')
        # TODO move to this? balance += m._activity.get_activity(comp, res, t)
        production = var[r, t]
      balance += production
  return balance == 0 # TODO tol?

def min_prod_rule(prod_name, r, caps, minimums, m, t) -> bool:
  """
    Constructs minimum production constraint
    @ In, prod_name, str, name of production variable
    @ In, r, int, index of resource for capacity constraining
    @ In, caps, list(float), capacity(t) value for component
    @ In, minimums, list(float), minimum allowable production in time
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, index of time variable
    @ Out, minimum, bool, min check
  """
  prod = getattr(m, prod_name)
  # negative capacity means consuming instead of producing
  if max(caps) > 0:
    return prod[r, t] >= minimums[t]
  else:
    return prod[r, t] <= minimums[t]

def ratio_transfer_rule(ratio: float, r: int, ref_r: int,prod_name: str, m, t) -> bool:
  """
    Constructs transfer function constraints
    @ In, ratio, float, balanced ratio of this resource to the reference resource
    @ In, r, int, index for this resource in the activity map
    @ In, ref_r, int, index of the reference resource in the activity map
    @ In, prod_name, str, name of production variable
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, index of time variable
    @ Out, transfer, bool, transfer ratio check
  """
  activity = getattr(m, prod_name)
  return activity[r, t] == activity[ref_r, t] * ratio

def poly_transfer_rule(coeffs, r_map, prod_name, m, t) -> bool:
  """
    Constructs transfer function constraints
    @ In, coeffs, dict, nested mapping of resources and polynomial orders to coefficients
          as {(r1, r2): {(o1, o2): n}}
    @ In, r_map, dict, mapping of resources to activity indices for this component
    @ In, prod_name, str, name of production variable
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, index of time variable
    @ Out, transfer, bool, transfer ratio check
  """
  activity = getattr(m, prod_name)
  eqn = 0
  for resources, ord_dict in coeffs.items():
    for orders, coeff in ord_dict.items():
      term = coeff
      for r, res in enumerate(resources):
        map_index = r_map[res]
        prod = activity[map_index, t]
        term *= prod ** orders[r]
      eqn += term
  return eqn == 0

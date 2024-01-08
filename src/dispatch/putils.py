"""
"""
import numpy as np
import pyomo.environ as pyo
from pprint import pprint

def get_all_resources(components):
  """
    Provides a set of all resources used among all components
    @ In, components, list, HERON component objects
    @ Out, resources, list, resources used in case
  """
  res = set()
  for comp in components:
    res.update(comp.get_resources())
  return res

def get_prod_bounds(m, comp, meta):
  """
    Determines the production limits of the given component
    @ In, comp, HERON component, component to get bounds of
    @ In, meta, dict, additional state information
    @ Out, lower, dict, lower production limit (might be negative for consumption)
    @ Out, upper, dict, upper production limit (might be 0 for consumption)
  """
  raise NotImplementedError
  # cap_res = comp.get_capacity_var()       # name of resource that defines capacity
  # r = m.resource_index_map[comp][cap_res]
  # maxs = []
  # mins = []
  # for t, time in enumerate(m.Times):
  #   meta['HERON']['time_index'] = t + m.time_offset
  #   cap = comp.get_capacity(meta)[0][cap_res]
  #   low = comp.get_minimum(meta)[0][cap_res]
  #   maxs.append(cap)
  #   if (comp.is_dispatchable() == 'fixed') or (low == cap):
  #     low = cap
  #     # initialize values to avoid boundary errors
  #     var = getattr(m, prod_name)
  #     values = var.get_values()
  #     for k in values:
  #       values[k] = cap
  #     var.set_values(values)
  #   mins.append(low)
  # maximum = comp.get_capacity(None, None, None, None)[0][cap_res]
  # # TODO minimum!
  # # producing or consuming the defining resource?
  # # -> put these in dictionaries so we can "get" limits or return None
  # if maximum > 0:
  #   lower = {r: 0}
  #   upper = {r: maximum}
  # else:
  #   lower = {r: maximum}
  #   upper = {r: 0}
  # return lower, upper

def get_transfer_coeffs(m, comp) -> dict:
  # FIXME DEPRECATE
  """
    Obtains transfer function ratios (assuming Linear ValuedParams)
    Form: 1A + 3B -> 2C + 4D
    Ratios are calculated with respect to first resource listed, so e.g. B = 3/1 * A
    TODO I think we can handle general external functions, maybe?
    @ In, m, pyo.ConcreteModel, associated model
    @ In, comp, HERON component, component to get coefficients of
    @ Out, ratios, dict, ratios of transfer function variables
  """
  transfer = comp.get_interaction().get_transfer()
  if transfer is None:
    return {}

  # linear transfer coefficients, dict as {resource: coeff}, SIGNS MATTER
  # it's all about ratios -> store as ratio of resource / first resource (arbitrary)
  coeffs = transfer.get_coefficients()
  coeffs_iter = iter(coeffs.items())
  first_name, first_coef = next(coeffs_iter)
  first_r = m.resource_index_map[comp][first_name]
  ratios = {'__reference': (first_r, first_name, first_coef)}

  for resource, coef in coeffs_iter:
    ratios[resource] = coef / first_coef

  return ratios

def retrieve_solution(m) -> dict:
  """
    Extracts solution from Pyomo optimization
    @ In, m, pyo.ConcreteModel, associated (solved) model
    @ Out, result, dict, {comp: {activity: {resource: [production]}} e.g. generator[production][steam]
  """
  return {
    component.name: {
      tag: retrieve_value_from_model(m, component, tag)
      for tag in component.get_tracking_vars()
    }
    for component in m.Components
  }

def retrieve_value_from_model(m, comp, tag) -> dict:
  """
    Retrieve values of a series from the pyomo model.
    @ In, m, pyo.ConcreteModel, associated (solved) model
    @ In, comp, Component, relevant component
    @ In, tag, str, particular history type to retrieve
    @ Out, result, dict, {resource: [array], etc}
  """
  result = {}
  prod = getattr(m, f'{comp.name}_{tag}')
  kind = 'Var' if isinstance(prod, pyo.Var) else 'Param'
  for res, comp_r in m.resource_index_map[comp].items():
    if kind == 'Var':
      result[res] = np.array([prod[comp_r, t].value for t in m.T], dtype=float)
    elif kind == 'Param':
      result[res] = np.array([prod[comp_r, t] for t in m.T], dtype=float)
  return result

### DEBUG
def debug_pyomo_print(m) -> None:
  """
    Prints the setup pieces of the Pyomo model
    @ In, m, pyo.ConcreteModel, model to interrogate
    @ Out, None
  """
  print('/' + '='*80)
  print('DEBUGG resource map:')
  pprint(m.resource_index_map)
  print('DEBUGG model pieces:')
  print('  -> objective:')
  print('     ', m.obj.pprint())
  print('  -> variables:')
  for var in m.component_objects(pyo.Var):
    print('     ', var.pprint())
  print('  -> constraints:')
  for constr in m.component_objects(pyo.Constraint):
    print('     ', constr.pprint())
  print('\\' + '='*80)
  print('')

def debug_print_soln(m) -> None:
  """
    Prints the solution from the Pyomo model
    @ In, m, pyo.ConcreteModel, model to interrogate
    @ Out, None
  """
  output = ['*' * 80, "DEBUGG solution:", f'  objective value: {m.obj()}']
  for c, comp in enumerate(m.Components):
    name = comp.name
    output.append(f'  component: {c} {name}')
    for tracker in comp.get_tracking_vars():
      prod = getattr(m, f'{name}_{tracker}')
      kind = 'Var' if isinstance(prod, pyo.Var) else 'Param'
      for res, r in m.resource_index_map[comp].items():
        output.append(f'    tracker: {tracker} resource {r}: {res}')
        for t, time in enumerate(m.Times):
          if kind == 'Var':
            value = prod[r, t].value
          elif kind == 'Param':
            value = prod[r, t]
          output.append(f'      time: {t + m.time_offset} {time} {value}')

  output.append('*' * 80)
  print('\n'.join(output))

# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Pyomo Utilities for Model Dispatch.
"""
import platform
import numpy as np
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError

def check_solver_availability(requested_solver: str) -> str:
  """
    Check if any of the requested solvers are available. If not, display available options.
    @ In, requested_solver, str, requested solver (e.g. 'cbc', 'glpk', 'ipopt')
    @ Out, solver, str, name of solver that is available to use.
  """
  # Choose solver; CBC is a great choice unless we're on Windows
  if platform.system() == 'Windows':
    platform_solvers = ['glpk', 'cbc', 'ipopt']
  else:
    platform_solvers = ['cbc', 'glpk', 'ipopt']

  solvers_to_check = platform_solvers if requested_solver is None else [requested_solver]
  for solver in solvers_to_check:
    if is_solver_available(solver):
      # Early return if everything is a-ok
      return solver

  # Otherwise raise an error
  all_options = pyo.SolverFactory._cls.keys()
  available_solvers = [op for op in all_options if not op.startswith('_') and is_solver_available(op)]
  raise RuntimeError(
    f'Requested solver "{requested_solver}" not found. Available options may include: {available_solvers}.'
  )

def is_solver_available(solver: str) -> bool:
  """
    Check if specified soler is available on the system.
    @ In, solver, str, name of solver to check.
    @ Out, is_available, bool, True if solver is available.
  """
  try:
    return pyo.SolverFactory(solver).available()
  except (ApplicationError, NameError, ImportError):
    return False

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


def get_initial_storage_levels(components: list, meta: dict, start_index: int) -> dict:
  """
      Return initial storage levels for 'Storage' component types.
      @ In, components, list, HERON components available to the dispatch.
      @ In, meta, dict, additional variables passed through.
      @ In, start_index, int, index of the start of the window.
      @ Out, initial_levels, dict, initial storage levels for 'Storage' component types.
  """
  initial_levels = {}
  for comp in components:
    if comp.get_interaction().is_type('Storage'):
      if start_index == 0:
        initial_levels[comp] = comp.get_interaction().get_initial_level(meta)
        # NOTE: There used to be an else conditional here that depended on the
        # variable `subdisp` which was not defined yet. Leaving an unreachable
        # branch of code, thus, I removed it. So currently, this function assumes
        # start_index will always be zero, otherwise it will return an empty dict.
        # Here was the line in case we need it in the future:
        # else: initial_levels[comp] = subdisp[comp.name]['level'][comp.get_interaction().get_resource()][-1]
  return initial_levels


def get_transfer_coeffs(m, comp) -> dict:
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

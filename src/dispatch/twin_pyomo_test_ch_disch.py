# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Enables quick testing of Pyomo problem solves, intending to set up a system
similar to how it is set up in the Pyomo dispatcher. This allows rapid testing
of different configurations for the rolling window optimization.
"""
import platform

import numpy as np

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

if platform.system() == 'Windows':
  SOLVER = 'glpk'
else:
  SOLVER = 'cbc'

# setup stuff
components = ['steam_source', 'elec_generator', 'steam_storage', 'elec_sink']
resources = ['steam', 'electricity']
time = np.linspace(0, 10, 11) # from @1 to @2 in @3 steps
dt = time[1] - time[0]
resource_map = {'steam_source': {'steam': 0},
                'elec_generator': {'steam': 0, 'electricity': 1},
                'steam_storage': {'steam': 0},
                'elec_sink': {'electricity': 0},
                }
activity = {}
for comp in components:
  activity[comp] = np.zeros((len(resources), len(time)), dtype=float)

# sizing specifications
storage_initial = 50 # kg of steam
storage_limit = 400 # kg of steam
steam_produced = 100 # kg/h of steam
gen_consume_limit = 110 # consumes at most 110 kg/h steam
sink_limit = 10000 # kWh/h = kW of electricity

def make_concrete_model():
  """
    Test writing a simple concrete model with terms typical to the pyomo dispatcher.
    @ In, None
    @ Out, m, pyo.ConcreteModel, instance of the model to solve
  """
  m = pyo.ConcreteModel()
  # indices
  C = np.arange(0, len(components), dtype=int) # indexes component
  R = np.arange(0, len(resources), dtype=int)  # indexes resources
  T = np.arange(0, len(time), dtype=int)       # indexes time
  # move onto model
  m.C = pyo.Set(initialize=C)
  m.R = pyo.Set(initialize=R)
  m.T = pyo.Set(initialize=T)
  # store some stuff for reference -> NOT NOTICED by Pyomo, we hope
  m.Times = time
  m.Components = components
  m.resource_index_map = resource_map
  m.Activity = activity
  #*******************
  #  set up optimization variables
  # -> for now we just do this manually
  # steam_source
  m.steam_source_index_map = pyo.Set(initialize=range(len(m.resource_index_map['steam_source'])))
  m.steam_source_production = pyo.Var(m.steam_source_index_map, m.T, initialize=steam_produced, bounds=(steam_produced, steam_produced))
  # elec_generator
  m.elec_generator_index_map = pyo.Set(initialize=range(len(m.resource_index_map['elec_generator'])))
  m.elec_generator_production = pyo.Var(m.elec_generator_index_map, m.T, initialize=0, bounds=(-gen_consume_limit, 0))
  # steam_storage
  m.steam_storage_index_map = pyo.Set(initialize=range(len(m.resource_index_map['steam_storage'])))
  m.steam_storage_level = pyo.Var(m.steam_storage_index_map, m.T, initialize=0, bounds=(0, storage_limit))
  m.steam_storage_charge = pyo.Var(m.steam_storage_index_map, m.T, initialize=0, bounds=(-storage_limit/dt, 0))
  m.steam_storage_discharge = pyo.Var(m.steam_storage_index_map, m.T, initialize=0, bounds=(0, storage_limit/dt))
  # elec_sink
  m.elec_sink_index_map = pyo.Set(initialize=range(len(m.resource_index_map['elec_sink'])))
  m.elec_sink_production = pyo.Var(m.elec_sink_index_map, m.T, initialize=0, bounds=(-sink_limit, 0))
  #*******************
  #  set up lower, upper bounds
  # -> for now we just do this manually
  # -> consuming is negative sign by convention!
  # -> producing is positive sign by convention!
  # steam source produces exactly 100 steam
  #    m.steam_source_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.steam_source_production[0, t] >= steam_produced)
  #    m.steam_source_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.steam_source_production[0, t] <= steam_produced)
  # elec generator can consume steam to produce electricity; 0 < consumed steam < 1000
  # -> this effectively limits electricity production, but we're defining capacity in steam terms for fun
  # -> therefore signs are negative, -1000 < consumed steam < 0!
  #    m.elec_generator_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.elec_generator_production[0, t] >= -gen_consume_limit)
  #    m.elec_generator_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.elec_generator_production[0, t] <= 0)
  # elec sink can take any amount of electricity
  # -> consuming, so -10000 < consumed elec < 0
  #    m.elec_sink_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.elec_sink_production[0, t] >= -sink_limit)
  #    m.elec_sink_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.elec_sink_production[0, t] <= 0)
  # storage is in LEVEL not ACTIVITY (e.g. kg not kg/s) -> lets say it can store X kg
  #    m.steam_storage_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.steam_storage_level[0, t] >= 0)
  #    m.steam_storage_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.steam_storage_level[0, t] <= storage_limit)
  #*******************
  # create transfer function
  # 2 steam make 1 electricity (sure, why not)
  m.elec_generator_transfer = pyo.Constraint(m.T, rule=_generator_transfer)
  #*******************
  # create conservation rules
  # steam
  m.steam_conservation = pyo.Constraint(m.T, rule=_conserve_steam)
  # electricity
  m.elec_conservation = pyo.Constraint(m.T, rule=_conserve_electricity)
  #*******************
  # create objective function
  m.OBJ = pyo.Objective(sense=pyo.maximize, rule=_economics)
  #######
  # return
  return m

#######
#
# Callback Functions
#
def _generator_transfer(m, t):
  """
    Constraint rule for electricity generation in generator
    @ In, m, pyo.ConcreteModel, model containing problem
    @ In, t, int, time indexer
    @ Out, constraint, bool, constraining evaluation
  """
  return - m.elec_generator_production[0, t] == 2.0 * m.elec_generator_production[1, t]

def _conserve_steam(m, t):
  """
    Constraint rule for conserving steam
    @ In, m, pyo.ConcreteModel, model containing problem
    @ In, t, int, time indexer
    @ Out, constraint, bool, constraining evaluation
  """
  # signs are tricky here, consumption is negative and production is positive
  # a positive delta in steam storage level means it absorbed steam, so it's a negative term
  storage_source = - (m.steam_storage_level[0, t] - (storage_initial if t == 0 else (m.steam_storage_charge[0, t-1] + m.steam_storage_discharge[0, t-1]))) / dt
  sources = storage_source + m.steam_source_production[0, t]
  sinks = m.elec_generator_production[0, t]
  return sources + sinks == 0

def _conserve_electricity(m, t):
  """
    Constraint rule for conserving electricity
    @ In, m, pyo.ConcreteModel, model containing problem
    @ In, t, int, time indexer
    @ Out, constraint, bool, constraining evaluation
  """
  sources = m.elec_generator_production[1, t]
  sinks = m.elec_sink_production[0, t]
  return sources + sinks == 0

def _economics(m):
  """
    Constraint rule for optimization target
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, objective, float, constraining evaluation
  """
  opex = sum(m.elec_generator_production[0, t] for t in m.T) * 10 # will be negative b/c consumed
  sales = - sum((m.elec_sink_production[0, t] * (100 if t < 5 else 1)) for t in m.T) # net positive because consumed
  return opex + sales

#######
#
# Debug printing functions
#
def print_setup(m):
  """
    Debug printing for pre-solve model setup
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, None
  """
  print('/' + '='*80)
  print('DEBUGG model pieces:')
  print('  -> objective:')
  print('     ', m.OBJ.pprint())
  print('  -> variables:')
  for var in m.component_objects(pyo.Var):
    print('     ', var.pprint())
  print('  -> constraints:')
  for constr in m.component_objects(pyo.Constraint):
    print('     ', constr.pprint())
  print('\\' + '='*80)
  print('')

def print_solution(m):
  """
    Debug printing for post-solve model setup
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, None
  """
  print('')
  print('*'*80)
  print('solution:')
  print('  objective value:', m.OBJ())
  print('time | steam source | storage level | elec gen (s, e) | elec sink')
  for t in m.T:
    print(f'{m.Times[t]:1.2e} | ' +
        f'{m.steam_source_production[0, t].value: 1.3e} | ' +
        f'{m.steam_storage_level[0, t].value: 1.3e} | ' +
        f'({m.elec_generator_production[0, t].value: 1.3e}, {m.elec_generator_production[1, t].value: 1.3e}) | ' +
        f'{m.elec_sink_production[0, t].value: 1.3e}'
        )
  print('*'*80)

#######
#
# Solver.
#
def solve_model(m):
  """
    Solves the model.
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, m, pyo.ConcreteModel, results
  """
  soln = pyo.SolverFactory(SOLVER).solve(m)
  return soln

if __name__ == '__main__':
  m = make_concrete_model()
  print_setup(m)
  s = solve_model(m)
  print_solution(m)

# solution using setup:
#   time = np.linspace(0, 10, 11)
#   storage_initial = 50 # kg of steam
#   storage_limit = 400 # kg of steam
#   steam_produced = 100 # kg/h of steam
#   gen_consume_limit = 110 # consumes at most 110 kg/h steam
#   sink_limit = 10000 # kWh/h = kW of electricity
#   1 steam = 2 * electricity
#   cost for generator = 10 * kg/h steam consumed
#   profit = 100 * electricity consumed at sink, t < 5,
#              1 * electricity consumed at sink, t >= 5,
#
# should look like:
# ********************************************************************************
# solution:
#   objective value: 20100.0
# time | steam source | steam storage | elec gen (s, e) | elec sink
# 0.00e+00 |  1.000e+02 |  4.000e+01 | (-1.100e+02,  5.500e+01) | -5.500e+01
# 1.00e+00 |  1.000e+02 |  3.000e+01 | (-1.100e+02,  5.500e+01) | -5.500e+01
# 2.00e+00 |  1.000e+02 |  2.000e+01 | (-1.100e+02,  5.500e+01) | -5.500e+01
# 3.00e+00 |  1.000e+02 |  1.000e+01 | (-1.100e+02,  5.500e+01) | -5.500e+01
# 4.00e+00 |  1.000e+02 |  0.000e+00 | (-1.100e+02,  5.500e+01) | -5.500e+01
# 5.00e+00 |  1.000e+02 |  0.000e+00 | (-1.000e+02,  5.000e+01) | -5.000e+01
# 6.00e+00 |  1.000e+02 |  0.000e+00 | (-1.000e+02,  5.000e+01) | -5.000e+01
# 7.00e+00 |  1.000e+02 |  1.000e+02 | ( 0.000e+00,  0.000e+00) |  0.000e+00
# 8.00e+00 |  1.000e+02 |  2.000e+02 | ( 0.000e+00,  0.000e+00) |  0.000e+00
# 9.00e+00 |  1.000e+02 |  3.000e+02 | ( 0.000e+00,  0.000e+00) |  0.000e+00
# 1.00e+01 |  1.000e+02 |  4.000e+02 | ( 0.000e+00,  0.000e+00) |  0.000e+00
# ********************************************************************************

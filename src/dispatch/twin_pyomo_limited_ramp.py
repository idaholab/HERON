# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Enables quick testing of Pyomo problem solves, intending to set up a system
similar to how it is set up in the Pyomo dispatcher. This allows rapid testing
of different configurations for the rolling window optimization.
"""
import platform # only for choosing solver based on HERON installation

import numpy as np
import matplotlib.pyplot as plt

import pyomo.environ as pyo

# This isn't strictly required, just how HERON installs default options
if platform.system() == 'Windows':
  SOLVER = 'glpk'
else:
  SOLVER = 'cbc'

# --------------------------------------------------------------------------------
# set up physical system, resources, time discretization
#
components = ['NPP', 'BOP', 'Grid']
resources = ['steam', 'electricity']
time = np.linspace(0, 24, 24) # from @1 to @2 in @3 steps
dt = time[1] - time[0]
resource_map = {'NPP': {'steam': 0},
                'BOP': {'steam': 0, 'electricity': 1},
                'Grid': {'electricity': 0},
                }

# sizing specifications
steam_produced = 100 # MWt/h of steam at NPP
gen_consume_limit = 110 # consumes at most 110 MWt steam at BOP
sink_limit = 1000 # MWh = MW of electricity at Grid

ramp_up_limit = 100 # MWt/h steam NPP ramp up limit
ramp_down_limit = 100 # MWt/h steam NPP ramp down limit
delta = 1 # aux var TODO good value?
time_between_ramp_up = 10 # hours before unit can ramp up again

# --------------------------------------------------------------------------------
# general economics
#
# marginal cost of NPP, 5 $/MWe = 1.67 $ / MWt
margin_NPP = 1.67
# sales price of electricity: use a sine function for demo
max_price = 10
freq = 2 * np.pi / 12
elec_price = max_price * np.sin(freq * time) ** 2
# debug plot of price curve
# import matplotlib.pyplot as plt
# plt.plot(time, elec_price, 'o-')
# plt.show()

# --------------------------------------------------------------------------------
# set up tracker for how components are dispatched ("activity")
# Map is activity: component: resource
# -> e.g. activity[NPP][steam]
#
activity = {}
for comp in components:
  activity[comp] = np.zeros((len(resources), len(time)), dtype=float)

# --------------------------------------------------------------------------------
# Set up method to construct the Pyomo concrete model.
#
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
  #*******************
  # FOCUS: create equations for limiting ramp frequency
  m.ramp_up = pyo.Var(m.T, initialize=0, within=pyo.Binary)
  m.ramp_down = pyo.Var(m.T, initialize=0, within=pyo.Binary)
  m.ramp_none = pyo.Var(m.T, initialize=0, within=pyo.Binary)
  # store some stuff for reference -> NOT NOTICED by Pyomo, we hope
  #*******************
  m.Times = time
  m.Components = components
  m.resource_index_map = resource_map
  m.Activity = activity
  #*******************
  #  set up optimization variables
  # -> for now we just do this manually
  # NPP
  m.NPP_index_map = pyo.Set(initialize=range(len(m.resource_index_map['NPP'])))
  m.NPP_production = pyo.Var(m.NPP_index_map, m.T, initialize=0)
  # BOP
  m.BOP_index_map = pyo.Set(initialize=range(len(m.resource_index_map['BOP'])))
  m.BOP_production = pyo.Var(m.BOP_index_map, m.T, initialize=0)
  # Grid
  m.Grid_index_map = pyo.Set(initialize=range(len(m.resource_index_map['Grid'])))
  m.Grid_production = pyo.Var(m.Grid_index_map, m.T, initialize=0)
  #*******************
  #  set up lower, upper bounds
  # -> for testing we just do this manually
  # -> consuming is negative sign by convention!
  # -> producing is positive sign by convention!
  # steam source produces between 0 and # steam
  m.NPP_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.NPP_production[0, t] >= 0)
  m.NPP_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.NPP_production[0, t] <= steam_produced)
  # elec generator can consume steam to produce electricity; 0 < consumed steam < 1000
  # -> this effectively limits electricity production, but we're defining capacity in steam terms for fun
  # -> therefore signs are negative, -1000 < consumed steam < 0!
  m.BOP_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.BOP_production[0, t] >= -gen_consume_limit)
  m.BOP_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.BOP_production[0, t] <= 0)
  # elec sink can take any amount of electricity
  # -> consuming, so -10000 < consumed elec < 0
  m.Grid_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.Grid_production[0, t] >= -sink_limit)
  m.Grid_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.Grid_production[0, t] <= 0)
  #*******************
  # create transfer function
  # 2 steam make 1 electricity (sure, why not)
  m.BOP_transfer = pyo.Constraint(m.T, rule=_generator_transfer)
  #*******************
  # create conservation rules
  # steam
  m.steam_conservation = pyo.Constraint(m.T, rule=_conserve_steam)
  # electricity
  m.elec_conservation = pyo.Constraint(m.T, rule=_conserve_electricity)
  #*******************
  # FOCUS: ramping rules
  m.npp_ramp_up_limit = pyo.Constraint(m.T, rule=_ramp_up_NPP)
  m.npp_ramp_down_limit = pyo.Constraint(m.T, rule=_ramp_down_NPP)
  m.npp_ramp_binaries = pyo.Constraint(m.T, rule=_ramp_binaries_NPP)
  m.npp_ramp_freq_limit = pyo.Constraint(m.T, rule=_ramp_time_limit_NPP)
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
  # 1 MWt steam -> 0.33 MWe power
  return - m.BOP_production[0, t] == 3.0 * m.BOP_production[1, t]

def _conserve_steam(m, t):
  """
    Constraint rule for conserving steam
    @ In, m, pyo.ConcreteModel, model containing problem
    @ In, t, int, time indexer
    @ Out, constraint, bool, constraining evaluation
  """
  # signs are tricky here, consumption is negative and production is positive
  sources = m.NPP_production[0, t]
  sinks = m.BOP_production[0, t]
  return sources + sinks == 0

def _conserve_electricity(m, t):
  """
    Constraint rule for conserving electricity
    @ In, m, pyo.ConcreteModel, model containing problem
    @ In, t, int, time indexer
    @ Out, constraint, bool, constraining evaluation
  """
  sources = m.BOP_production[1, t]
  sinks = m.Grid_production[0, t]
  return sources + sinks == 0

def _ramp_up_NPP(m, t):
  """
    Constraining rule for ramping up the NPP.
    @ In, m, pyomo.ConcreteModel, model
    @ In, t, int, relevant time index
    @ Out, rule, expression, evaluatable for Pyomo constraint
  """
  if t > 0:
    ineq = m.NPP_production[0, t] - m.NPP_production[0, t-1] <= ramp_up_limit * m.ramp_up[t] - delta * m.ramp_down[t]
  else:
    ineq = pyo.Constraint.Skip
  return ineq

def _ramp_down_NPP(m, t):
  """
    Constraining rule for ramping down the NPP.
    @ In, m, pyomo.ConcreteModel, model
    @ In, t, int, relevant time index
    @ Out, rule, expression, evaluatable for Pyomo constraint
  """
  if t > 0:
    ineq = m.NPP_production[0, t-1] - m.NPP_production[0, t] <= ramp_down_limit * m.ramp_down[t] - delta * m.ramp_up[t]
  else:
    ineq = pyo.Constraint.Skip
  return ineq

def _ramp_binaries_NPP(m, t):
  """
    Binaries to assure only one of (down, steady, up) can occur per time step
    @ In, m, pyomo.ConcreteModel, model
    @ In, t, int, relevant time index
    @ Out, rule, expression, evaluatable for Pyomo constraint
  """
  return m.ramp_up[t] + m.ramp_down[t] + m.ramp_none[t] == 1

def _ramp_time_limit_NPP(m, t):
  """
    Limits time between sequential ramp events
    @ In, m, pyomo.ConcreteModel, model
    @ In, t, int, relevant time index
    @ Out, rule, expression, evaluatable for Pyomo constraint
  """
  if t == 0:
    return pyo.Constraint.Skip
  tao = min(t, time_between_ramp_up)
  limit = 0
  for tm in range(t-tao, t):
    # equivalent and slightly easier to understand?
    limit += 1 - m.ramp_down[tm]
  return m.ramp_up[t] <= 1/tao * limit

def _economics(m):
  """
    Constraint rule for optimization target
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, objective, float, constraining evaluation
  """
  # marginal cost of operating NPP
  opex = sum(m.BOP_production[0, t] for t in m.T) * margin_NPP # will be negative b/c consumed
  # grid sales, calculated above; negative sign because we're consuming
  sales = - sum((m.Grid_production[0, t] * elec_price[t]) for t in m.T) # net positive because consumed
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

def extract_soln(m):
  """
    Extracts final solution from model evaluation
    @ In, m, pyo.ConcreteModel, model
    @ Out, res, dict, results dictionary for dispatch
  """
  res = {}
  T = len(m.T)
  res['prices'] = np.zeros(T)
  res['NPP_steam'] = np.zeros(T)
  res['BOP_steam'] = np.zeros(T)
  res['BOP_elec'] = np.zeros(T)
  res['Grid_elec'] = np.zeros(T)
  for t in m.T:
    res['prices'][t] = elec_price[t]
    res['NPP_steam'][t] = m.NPP_production[0, t].value
    res['BOP_steam'][t] = m.BOP_production[0, t].value
    res['BOP_elec'][t] = m.BOP_production[1, t].value
    res['Grid_elec'][t] = m.Grid_production[0, t].value
  return res

def plot_solution(m):
  """
    Plots solution from optimized model
    @ In, m, pyo.ConcreteModel, model
    @ Out, None
  """
  res = extract_soln(m)
  fig, axs = plt.subplots(3, 1, sharex=True)
  axs[0].set_ylabel(r'Steam MW$_t$')
  axs[1].set_ylabel(r'Elec MW$_e$')
  axs[2].set_ylabel(r'Prices \$/MW$_e$')
  axs[2].set_xlabel('Time (h)')
  axs[0].plot(time, res['NPP_steam'], 'o-', label='NPP_steam')
  axs[0].plot(time, res['BOP_steam'], 'o-', label='BOP_steam')
  axs[0].legend()
  axs[1].plot(time, res['BOP_elec'], 'o-', label='BOP_elec')
  axs[1].plot(time, res['Grid_elec'], 'o-', label='Grid_elec')
  axs[1].legend()
  axs[2].plot(time, res['prices'], 'o-', label='Prices')
  axs[2].legend()
  plt.suptitle(f'Ramp up limit: {time_between_ramp_up} h')
  plt.savefig(f'dispatch_limit_{time_between_ramp_up}.png')

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
  print('time     |  price | steam src  |     elec gen (s, e)      | elec sink')
  for t in m.T:
    print(f'{m.Times[t]:1.2e} | ' +
        f'{elec_price[t]: 3.3f} | ' +
        f'{m.NPP_production[0, t].value: 1.3e} | ' +
        f'({m.BOP_production[0, t].value: 1.3e}, {m.BOP_production[1, t].value: 1.3e}) | ' +
        f'{m.Grid_production[0, t].value: 1.3e}'
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
  plot_solution(m)
  plt.show()

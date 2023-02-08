# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

import pyomo.environ as pyo
import numpy as np
"""
  Implements transfer functions
"""

def storage_dependent_efficiency(data, meta):
  """
    Adds constraint for steam-to-electricty transfer based on storage level.

    NOTE: the information within `data` should be in a specific format. User must specify what
    data type is being returned in the `ratio_type` entry:
      - data['ratio_type'] : str, could be either 'lists' **or** 'custom_pyomo'
    Then, in an entry of the same name, the user must report back the values in a nested dictionary.
    For the `custom_pyomo` option, the nested dictionary should look like:
      * data['custom_pyomo']['rule_name'] = <Pyomo Constraint>

    @ In, data, dict, information to be filled before return
    @ In, meta, dict, additional information from HERON state
    @ Out, data, dict, information filled
    @ Out, meta, dict, additional information from HERON state
  """
  comp = data['component']
  # time = data['time']
  T_index = data['T_index']
  r_index = data['r_index']
  rule_name_template = data['rule_name_template']
  rule_name = rule_name_template(comp.name, 'steam', 'electricity')

  storage_comp_name = 'steam_storage'
  storage = [meta['HERON']['Components'][c] for c, comp in enumerate(meta['HERON']['Components']) if comp.name == storage_comp_name][0]
  stor_name = f'{storage_comp_name}_charge'
  r_stor = r_index[storage]['steam']

  prod_name = f'{comp.name}_production'
  r_prod = r_index[comp]['electricity']
  r_ref  = r_index[comp]['steam']
  efficiency = -0.5
  rule = lambda mod, t: transfer_storage_rule(efficiency, r_prod, r_ref, r_stor, prod_name, stor_name, mod, t)
  constr = pyo.Constraint(T_index, rule=rule)

  transfers = {}
  transfers['ratio_type'] = 'custom_pyomo'
  transfers['custom_pyomo'] = {}
  transfers['custom_pyomo'][rule_name] = constr

  data = {'transfers': transfers}

  return data, meta

def transfer_storage_rule(eff, r, ref_r, r_stor, prod_name, stor_name, m, t):
  """
    Constructs transfer function constraints
    @ In, r, int, index of transfer resource
    @ In, ref_r, int, index of reference resource
    @ In, prod_name, str, name of production variable
    @ In, m, pyo.ConcreteModel, associated model
    @ In, t, int, index of time variable
    @ Out, transfer, bool, transfer ratio check
  """
  prod = getattr(m, prod_name)
  stor = getattr(m, stor_name)
  # implementing a fake rule where efficiency drops if tank is charging
  if pyo.value(stor[r_stor, t]) > 0:
    return prod[r, t] == prod[ref_r, t] * eff * 0.6
  return prod[r, t] == prod[ref_r, t] * eff

# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

import pyomo.environ as pyo
import numpy as np
"""
  Implements transfer functions
"""

# some global variables
MULT = 25
INTCPT = 70

def temp_dependent_efficiency(data, meta):
  """
    Determines transfer coefficient between steam and electricity (i.e. efficiency) as a time series
    NOTE: the information within `data` should be in a specific format. User must specify what
    data type is being returned in the `ratio_type` entry:
      - data['ratio_type'] : str, could be either 'lists' **or** 'custom_pyomo'
    Then, in an entry of the same name, the user must report back the values in a nested dictionary.
    For the `lists` option, the nested dictionary should look like:
        * data['lists']['__reference'] = <name_of_reference_resource>
        * data['lists']['<other_resource>'] = <list_of_ratios_indexed_by_time>

    @ In, data, dict, information to be filled before return
    @ In, meta, dict, additional information from HERON state
    @ Out, data, dict, information filled
    @ Out, meta, dict, additional information from HERON state
  """
  # comp = data['component']
  time = data['time']
  # T_index = data['T_index']
  # r_index = data['r_index'][comp]
  # rule_name_template = data['rule_name_template']

  temperature_signal = meta['HERON']['RAVEN_vars']['Signal']
  temperature = temperature_signal*MULT + INTCPT

  transfers = {}
  transfers['ratio_type'] = 'lists'
  transfers['lists'] = {}
  transfers['lists']['__reference'] = 'steam'
  transfers['lists']['electricity'] = np.zeros(len(time))
  for t, _ in enumerate(time):
    if temperature[t] > 80:
      # steam to electricity efficiency lower at higher temps
      eff = -0.4
    else:
      eff = -0.5
    transfers['lists']['electricity'][t] = eff

  data = {'transfers': transfers}

  return data, meta

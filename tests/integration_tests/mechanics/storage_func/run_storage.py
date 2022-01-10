
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements storage control
"""
import numpy as np

def tiered(data, meta):
  """
    Controls storage based on level and demand
    @ In, data, dict, information to be filled before return
    @ In, meta, dict, additional information from HERON state
    @ Out, data, dict, information filled
    @ Out, meta, dict, additional information from HERON state
  """
  # very simplistic example
  comp = data['component']
  time = meta['HERON']['RAVEN_vars']['Time']
  dt = time[1] - time[0]
  price = meta['HERON']['RAVEN_vars']['Signal']
  steam_produced = meta['HERON']['RAVEN_vars']['steamer_capacity'][0] * dt
  capacity = comp.get_capacity(meta)[0]['steam']
  generator = [meta['HERON']['Components'][c] for c, comp in enumerate(meta['HERON']['Components']) if comp.name == 'generator'][0]
  gen_cap = generator.get_capacity(meta)[0]['steam'] * dt * -1
  current_level = comp.get_interaction().get_initial_level(meta)
  levels = np.zeros(len(time))
  for t, _ in enumerate(time):
    if price[t] > 0.7:
      # empty storage and sell it all, up to generator capacity
      release = min(current_level, gen_cap - steam_produced)
      current_level -= release
    elif price[t] < 0.3:
      # take and store all the electricity we can
      absorb = min(steam_produced, capacity - current_level)
      current_level += absorb
    levels[t] = current_level
  data['level'] = levels
  return data, meta

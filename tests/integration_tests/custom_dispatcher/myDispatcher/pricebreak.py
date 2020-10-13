# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Example of a custom dispatching strategy
"""
import numpy as np

w_price = 0.05 # $/water/s
w_per_e = 10  # 10 water / elec
w_eff_price = w_price * w_per_e # $/MW, when e is used to make water

def dispatch(meta):
  # find price
  e_price_hist = meta['HERON']['RAVEN_vars']['Signal']
  T = len(e_price_hist)
  # find out how much electricity we're producing
  ## same as the capacity of the power plant
  for comp in meta['HERON']['Components']:
    if comp.name == 'power_plant':
      power_plant = comp
      break
  electricity_avail = power_plant.get_capacity(meta)[0]['electricity']
  # choose how to dispatch units (define their activity)
  ## activity is: per component, per resource, vector of activity in time
  activity = {
    # power plant always makes all of its electricity
    'power_plant': {'electricity': electricity_avail * np.ones(T)},
    # other units we initialize at 0 for the resources they deal with
    'desal': {'water': np.zeros(T),
              'electricity': np.zeros(T)},
    'e_market': {'electricity': np.zeros(T)},
    'w_market': {'water': np.zeros(T)}
  }
  # numpy mask for when water price is high
  choose_e_mask = e_price_hist >= w_eff_price
  choose_w_mask = np.logical_not(choose_e_mask)
  # when water is favorable, use all electricity at the desal (negative is consuming)
  activity['desal']['electricity'][choose_w_mask] = - electricity_avail
  # water produced is electricity used * transfer rate
  activity['desal']['water'][choose_w_mask] = electricity_avail * w_per_e
  # all water produced is consumed at the water market (negative is conuming)
  activity['w_market']['water'] = - activity['desal']['water']
  # all other electricity is sold at the grid (negative is conusming)
  activity['e_market']['electricity'][choose_e_mask] = - electricity_avail
  return activity



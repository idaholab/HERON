# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Example of a custom dispatching strategy
"""
import numpy as np

w_price = 0.05 # $/water/s
w_per_e = 10  # 10 water / elec
w_eff_price = w_price * w_per_e # $/MW, when e is used to make water

def dispatch(info):
  """
    Dispatches the components based on user-defined algorithms.
    The expected return object is a dict of components, mapped to a dict of
      resources that component uses, mapped to the amount consumed/produced
      as a numpy array.
    Note:
     - Negative values mean the component consumes that resource
     - Positive values mean the component produces that resource
     - The activity doesn't necessarily have to be as long as "time", but it usually should be
    @ In, info, dict, information about the state of the system
    @ Out, activity, dict, activity of components as described above
  """
  # find price
  e_price_hist = info['HERON']['RAVEN_vars']['Signal']
  T = len(e_price_hist)
  # find out how much electricity we're producing
  ## same as the capacity of the power plant
  for comp in info['HERON']['Components']:
    if comp.name == 'power_plant':
      power_plant = comp
      break
  electricity_avail = power_plant.get_capacity(info)[0]['electricity']
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




# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  ## works with generic?
  # activity = meta['raven_vars']['HERON_pyomo_model']
  # t = meta['t']
  # flip sign because we consume the electricity
  # E = -1.0 * activity['electricity'][t]

  ## works with pyomo
  # model = meta['HERON']['pyomo_model']
  # component = meta['HERON']['component']
  activity = meta['HERON']['activity']
  #print("This is activity", activity, activity.keys(),activity['electricity'])
  #aaa
  #print("This is activity", activity, activity.keys())
  # TODO a get_activity method for the dispatcher -> returns object-safe activity (expression or value)?
  amount = -1 * activity['electricity']
  data = {'driver': amount}
  return data, meta

def flex_price(data, meta):
  #print(meta['HERON'].keys())
  #print("This is the activity", meta['HERON']['activity']['electricity'])
  #aaa
  sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  #print("This is signal", sine)
  #aaa
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount = - 2 * (sine[t] - 0.5)
  data = {'reference_price': amount}
  return data, meta

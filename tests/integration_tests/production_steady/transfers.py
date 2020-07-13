
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
  # TODO a get_activity method for the dispatcher -> returns object-safe activity (expression or value)?
  E = -1 * activity['electricity']
  data = {'driver': E}
  return data, meta

"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  # flip sign because we consume the electricity
  activity = meta['raven_vars']['HERON_activity_report']
  t = meta['t']
  E = -1.0 * activity['electricity'][t]
  data = {'driver': E}
  return data, meta

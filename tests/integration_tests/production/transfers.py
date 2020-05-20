"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  # flip sign because we consume the electricity
  E = -1.0 * meta['raven_vars']['electricity']
  data = {'driver': E}
  return data, meta

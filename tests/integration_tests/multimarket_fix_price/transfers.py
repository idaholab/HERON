"""
  Implements transfer functions
"""

def electric_source(data, meta):
  """
    Fix up electricity for how much the Grids consume
  """
  # flip sign because we consume the electricity
  E = -1.0 * meta['raven_vars']['electricity']
  data = {'driver': E}
  return data, meta

def power_conversion(data, meta):
  """
    How to get power from the incoming signal
  """
  # get the signal (year, time) from RAVEN ARMA
  signal = meta['raven_vars']['Signal'][:, :]
  # what time step are we currently at?
  index = meta['t']
  # boost the signal's mean uniformly, just because
  signal += 10
  # return the entry from the appropriate index
  power = signal[index, 0]
  # set the value to return
  data['electricity'] = power
  return(data, meta)
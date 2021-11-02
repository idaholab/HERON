
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def power_conversion(data, meta):
  """
    How to get power from the incoming signal
  """
  # get the signal (year, time) from RAVEN ARMA
  ## NOTE this behaves completely different if you remove
  # the 1.0, and I have no idea why. Leave it there, and
  # you get the correct analytic results.
  signal = 1.0 * meta['raven_vars']['Signal'][:, :]
  # what time step are we currently at?
  index = meta['HERON']['time_index']
  # return the entry from the appropriate index
  power = signal[index, 0] + 10
  # set the value to return
  data['electricity'] = power
  return(data, meta)

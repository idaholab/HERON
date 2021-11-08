
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def flex_price(data, meta):
  """
    Transfer function to acquire the flexible price of electricity
    @ In, data, dict, partial requested activity
    @ In, meta, dict, additional info
    @ Out, data, dict, filled-in requested activity
    @ Out, meta, dict, additional info (possibly modified, possibly not)
  """
  sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount = - 2 * (sine[t] - 0.5)
  data = {'reference_price': amount}
  return data, meta

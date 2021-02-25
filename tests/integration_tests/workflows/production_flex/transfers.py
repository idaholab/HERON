
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  """
    Provides the amount of electricity consumed.
    @ In, data, dict, request for data
    @ In, meta, dict, state information
    @ Out, data, dict, filled data
    @ In, meta, dict, state information
  """
  activity = meta['HERON']['activity']
  # TODO a get_activity method for the dispatcher -> returns object-safe activity (expression or value)?
  amount = activity['electricity']
  # NOTE multiplier is -1 in the input!
  data = {'driver': amount}
  return data, meta

def flex_price(data, meta):
  """
    Determines the price of electricity.
    @ In, data, dict, request for data
    @ In, meta, dict, state information
    @ Out, data, dict, filled data
    @ In, meta, dict, state information
  """
  sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  # scale electricity consumed to flex between -1 and 1
  amount = - 2 * (sine[t] - 0.5)
  data = {'reference_price': amount}
  return data, meta

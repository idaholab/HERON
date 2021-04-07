
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def energy_consume(data, meta):
  """
    Provides the amount of energy consumed.
    @ In, data, dict, request for data
    @ In, meta, dict, state information
    @ Out, data, dict, filled data
    @ In, meta, dict, state information
  """
  activity = meta['HERON']['activity']
  amount = activity['energy']
  data = {'driver': amount}
  return data, meta

def commodity_consume(data, meta):
  """
    Provides the amount of commodity consumed.
    @ In, data, dict, request for data
    @ In, meta, dict, state information
    @ Out, data, dict, filled data
    @ In, meta, dict, state information
  """
  activity = meta['HERON']['activity']
  amount = activity['commodity']
  data = {'driver': amount}
  return data, meta

# def flex_price(data, meta):
#   """
#     Determines the price of electricity.
#     @ In, data, dict, request for data
#     @ In, meta, dict, state information
#     @ Out, data, dict, filled data
#     @ In, meta, dict, state information
#   """
#   sine = meta['HERON']['RAVEN_vars']['Signal']
#   t = meta['HERON']['time_index']
#   # scale electricity consumed to flex between -1 and 1
#   amount = - 2 * (sine[t] - 0.5)
#   data = {'reference_price': amount}
#   return data, meta

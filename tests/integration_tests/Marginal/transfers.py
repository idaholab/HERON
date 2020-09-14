
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  
  activity = meta['HERON']['activity']

  # TODO a get_activity method for the dispatcher -> returns object-safe activity (expression or value)?
  amount = -1 * activity['electricity']
  data = {'driver': amount}
  return data, meta

def flex_price(data, meta):
  #print("This is HERON", meta)
  #aaa
  sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount = - 2 * (sine[t] - 0.5)
  data = {'reference_price': amount}
  return data, meta

def H2_consume(data, meta):
  activity = meta['HERON']['activity']
  amount = -1 * activity['hydrogen']
  #t = meta['HERON']['time_index']
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  #amount = - 2 * (sine[t] - 0.5)
  data = {'driver': amount}
  return data, meta

def steam_consume(data, meta):

  activity = meta['HERON']['activity']

  # TODO a get_activity method for the dispatcher -> returns object-safe activity (expression or value)?
  amount = -1 * activity['steam']
  data = {'driver': amount}
  return data, meta

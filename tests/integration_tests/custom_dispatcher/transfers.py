
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  """
    Changes sign of electricity consumed
    @ In, data, dict, info
    @ In, meta, dict, info
    @ Out, data, dict, info
    @ Out, meta, dict, info
  """
  activity = meta['HERON']['activity']
  amount = -1 * activity['electricity']
  data = {'driver': amount}
  return data, meta

def water_consume(data, meta):
  """
    Changes sign of water consumed
    @ In, data, dict, info
    @ In, meta, dict, info
    @ Out, data, dict, info
    @ Out, meta, dict, info
  """
  activity = meta['HERON']['activity']
  amount = -1 * activity['water']
  data = {'driver': amount}
  return data, meta

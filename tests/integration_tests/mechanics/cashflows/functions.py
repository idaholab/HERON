# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def compute_price(data, meta):
  data = {'reference_price': -10000.0}
  return data, meta

def capacity(data, meta):
  """
    return unit capacity
    @ In, data, dict, data requeset
    @ In, meta, dict, state information
    @ Out, data, dict, filled data
    @ Out, meta, dict, state information
  """
  c = float(meta['HERON']['RAVEN_vars']['source_capacity'])
  data = {'driver': c}
  return data, meta


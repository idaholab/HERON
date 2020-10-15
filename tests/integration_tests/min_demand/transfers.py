# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def electric_source(data, meta):
  """
    Fix up electricity for how much the Grids consume
    @ In, data, dict, data requeset
    @ In, meta, dict, state information
    @ Out, data, dict, filled data
    @ Out, meta, dict, state information
  """
  # flip sign because we consume the electricity
  E = -1.0 * meta['HERON']['activity']['electricity']
  data = {'driver': E}
  return data, meta


# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""
def electric_source(data, meta):
  """
    Fix up electricity for how much the Grids consume
  """
  # flip sign because we consume the electricity
  E = -1.0 * meta['HERON']['activity']['electricity']
  data = {'driver': E}
  return data, meta


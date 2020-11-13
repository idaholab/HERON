
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Class for mimicking ARMA structure
"""

import numpy as np

def run(raven, raven_dict):
  """
    Makes ARMA-like data structure, same guy every time.
    @ In, raven, object, RAVEN variables object
    @ In, raven_dict, dict, additional RAVEN information
    @ Out, None
  """
  T = 24
  Y = 26
  C = 20
  data = np.ones((Y, C, T))
  raven.TOTALLOAD = data
  im = getattr(raven, '_indexMap', [{}])[0]
  im['TOTALLOAD'] = ['YEAR', '_ROM_Cluster', 'HOUR']
  raven._indexMap = np.atleast_1d([im])
  raven.YEAR = np.arange(Y) + 2025
  raven._ROM_Cluster = np.arange(C)
  raven.HOUR = np.arange(T)





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
  # take some time working hard
  for i in range(int(1e5)):
    x = 2**i
  # result
  raven.NPV = np.random.rand()
  raven.time_delta = 3.14159




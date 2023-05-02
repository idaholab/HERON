import numpy as np
"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  """
    Swaps the sign for electricity consumption to yield positive cash values.
    @ In, data, dict, information to be filled before return
    @ In, meta, dict, additional information from HERON state
    @ Out, data, dict, information filled
    @ Out, meta, dict, additional information from HERON state
  """
  activity = meta['HERON']['activity']
  amount = -1 * activity['electricity']
  data = {'driver': amount}
  return data, meta

def tes(input, init_level):
    '''This is a storage component transfer function. This transfer function
    works just the same as the others with two significant differences.
    1) It requires a second argument that is the intial storage level of the
    component.
    2) Instead of returning the activities of the other involved
    resources (there are none for storage components) it returns a time-
    resolved array of storage levels for the component.'''
    tmp = np.insert(input, 0, init_level)
    return np.cumsum(tmp)[1:]

def flex_price(data, meta):
  """
    Gathers and modifies the ARMA signal to produce a price history ranging
    from -1 to 1 instead of 0 to 1.
    @ In, data, dict, information to be filled before return
    @ In, meta, dict, additional information from HERON state
    @ Out, data, dict, information filled
    @ Out, meta, dict, additional information from HERON state
  """
  sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount = - 2 * (sine[t] - 0.5)
  data = {'reference_price': amount}
  return data, meta

def generator(inputs):
  return {'steam': -1 / 0.7 * inputs}


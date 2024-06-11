
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

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
  # check for the existince of the custom functions
  for node in meta['HERON']['custom_input']:
    if node.tag == 'General':
      ans = node.find('TheAnswer').text
      qus = node.find('TheQuestion').text
    elif node.tag == 'scalar':
      scalar = float(node.text)
    elif node.tag == 'loc':
      loc = float(node.text)
  print(f'The answer to the question "{qus}" is "{ans}".')
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount = scalar * (sine[t] + loc)
  data = {'reference_price': amount}
  return data, meta

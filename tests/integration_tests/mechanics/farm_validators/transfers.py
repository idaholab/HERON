
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""
import math

def electric_consume(data, meta):
  """
    Transfer function to change sign of electricity consumed, for cashflows
    @ In, data, dict, partial requested activity
    @ In, meta, dict, additional info
    @ Out, data, dict, filled-in requested activity
    @ Out, meta, dict, additional info (possibly modified, possibly not)
  """
  activity = meta['HERON']['activity']
  amount = -1 * activity['production']['electricity']
  data = {'driver': amount}
  return data, meta

def flex_price(data, meta):
  """
    Transfer function to acquire the flexible price of electricity
    @ In, data, dict, partial requested activity
    @ In, meta, dict, additional info
    @ Out, data, dict, filled-in requested activity
    @ Out, meta, dict, additional info (possibly modified, possibly not)
  """
  sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount = - 2 * (sine[t] - 0.5)
  data = {'reference_price': amount}
  return data, meta

def electric_prod(data, meta):
  """
    Transfer function to change sign of electricity consumed, for cashflows
    @ In, data, dict, partial requested activity
    @ In, meta, dict, additional info
    @ Out, data, dict, filled-in requested activity
    @ Out, meta, dict, additional info (possibly modified, possibly not)
  """
  activity = meta['HERON']['activity']
  amount = 1 * activity['production']['electricity']
  data = {'driver': amount}
  return data, meta

def flex_cost(data, meta):
  """
    Transfer function to acquire the flexible price of electricity
    @ In, data, dict, partial requested activity
    @ In, meta, dict, additional info
    @ Out, data, dict, filled-in requested activity
    @ Out, meta, dict, additional info (possibly modified, possibly not)
  """
  sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  # Heron = meta
  # print("Haoyu Debug, meta=",Heron)
  # activity = meta['HERON']['activity']
  # print("Haoyu Debug, activity=",activity)
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount =  100*(sine[t]-1)-50
  data = {'reference_price': amount}
  print("Haoyu Debug, t={}, flex_cost amount={}".format(t,amount))
  return data, meta

def flex_demand(data, meta):
  """
    Transfer function to acquire the flexible price of electricity
    @ In, data, dict, partial requested activity
    @ In, meta, dict, additional info
    @ Out, data, dict, filled-in requested activity
    @ Out, meta, dict, additional info (possibly modified, possibly not)
  """
  # sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  # Heron = meta
  # print("Haoyu Debug, meta=",Heron)
  # activity = meta['HERON']['activity']
  # print("Haoyu Debug, activity=",activity)
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount =  55./2*(math.sin(math.pi*t)-1) - 1485.
  data = {'demands': amount}
  print("Haoyu Debug, t={}, flex_demand amount={}".format(t,amount))
  return data, meta

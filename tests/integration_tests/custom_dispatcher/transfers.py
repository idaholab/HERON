
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  activity = meta['HERON']['activity']
  amount = -1 * activity['electricity']
  data = {'driver': amount}
  return data, meta

def water_consume(data, meta):
  activity = meta['HERON']['activity']
  amount = -1 * activity['water']
  data = {'driver': amount}
  return data, meta

# def e_price(data, meta):
#   signal = meta['HERON']['RAVEN_vars']['Signal']
#   data = {'reference_price': sine}
#   return data, meta

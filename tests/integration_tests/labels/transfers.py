
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Implements transfer functions
"""

def electric_consume(data, meta):
  # works with pyomo
  activity = meta['HERON']['activity']
  amount = -1 * activity['electricity']
  data = {'driver': amount}
  return data, meta,

def flex_price(data, meta):
  sine = meta['HERON']['RAVEN_vars']['Signal']
  t = meta['HERON']['time_index']
  # DispatchManager
  # scale electricity consumed to flex between -1 and 1
  amount = - 2 * (sine[t] - 0.5)
  labels = meta['HERON']['Case'].get_labels()
  pivot_id = meta['HERON']['Case'].get_time_name()
  year_id = meta['HERON']['Case'].get_year_name()
  data = {'reference_price': amount,
          'case_labels': labels,
          'pivot_id': pivot_id,
          'year_id': year_id}
  return data, meta

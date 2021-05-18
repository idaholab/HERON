# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

def run(raven, inputs):
  """
    Example model for an arithmatic evaluation.
    @ In, raven, object, variable entity from RAVEN
    @ In, inputs, dict, additional inputs
    @ Out, None
  """
  a = raven.a
  b = raven.b
  c = raven.c
  d = raven.d
  raven.price = a + 2*b - 3*c + 0.5*d

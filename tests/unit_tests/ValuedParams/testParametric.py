# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Test specific aspects of HERON Parametric ValuedParams
"""

import os
import sys
import xml.etree.ElementTree as ET

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir]*4))
sys.path.append(HERON_LOC)

from HERON.src.ValuedParams import factory
from HERON.src import ValuedParams
from HERON.src import _utils as hutils
sys.path.pop()

try:
  import ravenframework
except ModuleNotFoundError:
  # Load RAVEN tools
  sys.path.append(hutils.get_raven_loc())
from ravenframework.utils import InputData, xmlUtils,InputTypes
import ravenframework.MessageHandler

results = {"pass":0, "fail":0}

# test retrieving classes
for alias in ['fixed_value', 'sweep_values', 'opt_bounds']:
  kls = factory.returnClass(alias)
  if issubclass(kls, ValuedParams.Parametric):
    results['pass'] += 1
  else:
    results['fail'] += 1
    print(f'Alias "{alias}" did not return "ValuedParams.Parametric"!')

##################
#
# Parametric
#
p = factory.returnInstance('fixed_value')
#
# skip reading
#
#
# set value
#
expect = 3.14
p.set_value(expect)
# get directly
get = p.get_value()
if get == expect:
  results['pass'] += 1
else:
  results['fail'] += 1
  print(f'Parametric value set and get was not the same: set {expect} but got {get}!')
# evaluate
val = p.evaluate(None, target_var='custom_target')[0]['custom_target']
if val == expect:
  results['pass'] += 1
else:
  results['fail'] += 1
  print(f'Parametric value set and get was not the same: set {expect} but got {val}!')

print(results)
sys.exit(results['fail'])

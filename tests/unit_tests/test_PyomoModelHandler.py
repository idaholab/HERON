'''
  Test specific aspects of HERON Pyomo dispatcher's PyomoModeHandler
'''

import os
import sys
import pytest
import xml.etree.ElementTree as ET

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(HERON_LOC)
from HERON.src import Components
from HERON.src import _utils as hutils
from HERON.src.ValuedParamHandler import ValuedParamHandler as VPH
from HERON.src.ValuedParams import factory as VPfactory
from HERON.src.dispatch.PyomoModelHandler import PyomoModelHandler as PMH
pmh = PMH(0, 0, None, None, None, None, None)
sys.path.pop()

import pyomo.environ as pyo
try:
  import ravenframework
except ModuleNotFoundError:
    # Load RAVEN tools
    sys.path.append(hutils.get_raven_loc())
#from ravenframework.utils import InputData, xmlUtils,InputTypes
import ravenframework.MessageHandler as MessageHandler


@pytest.fixture
def component():
  transfer = VPfactory.returnInstance('poly')
  # encode af^2 + bf + cf^0.2 + k_1 = dw + k_2
  # becomes af^2 + bf + cf^0.2 + k - dw = 0
  # divide out d for new a, b, c, k
  #         af^2 + bf + cf^0.2 + k - w = 0
  # where:
  #  f is "funding" in [0,400],
  #  w is "work" in ~ 37 when x~14.5
  #  a, b, c, d, are all scalar coefficients:
  #    a =  1e-3
  #    b = -0.5
  #    c = 20
  #    k = 10
  coeffs = {
    ('funding'): {(2): 1e-3, (1): -0.5, (0.2): 20, (0): 10}
  }
  transfer._coefficients = coeffs
  comp = Components.Producer()
  comp.name = 'postdoc'
  comp.messageHandler = MessageHandler.MessageHandler()
  comp.messageHandler.verbosity = 'debug'
  comp._capacity_var = 'funding'
  comp._capacity = VPH('researcher_capacity')
  comp._capacity.set_const_VP(0)
  comp.set_capacity(400)
  comp._transfer = transfer
  yield comp

class TestHeronPMH:

  def test_prod_tracker(self, component):
    assert component.get_tracking_vars() == ['production']

  def test_transfer(self, component):
    m = pyo.ConcreteModel()
    pmh._create_transfer(m, component)
    print(m)
    assert False


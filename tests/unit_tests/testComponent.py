'''
Test specific aspects of HERON Components
'''

import os
import sys
import xml.etree.ElementTree as ET

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(HERON_LOC)
from HERON.src import Components
from HERON.src.ValuedParamHandler import ValuedParamHandler
from HERON.src import _utils as hutils
sys.path.pop()

try:
  import ravenframework
except ModuleNotFoundError:
    # Load RAVEN tools
    sys.path.append(hutils.get_raven_loc())
from ravenframework.utils import InputData, xmlUtils,InputTypes
import ravenframework.MessageHandler as MessageHandler

results = {"pass":0,"fail":0}

#----------------------------------------
# Test Producer
#

# Set up the dummy transfer function
def transfer_function(inputs): #method, requests, inputs):
    request = inputs.pop('request', None)
    # Return the given request and the given meta (inputs)
    return {'electricity': 0}, inputs

# Set up the component
producer = Components.Producer()
producer.messageHandler = MessageHandler.MessageHandler()
producer.messageHandler.verbosity = 'debug'
# these are usually set in reading the input
producer._capacity_var = 'electricity'
producer._capacity = ValuedParamHandler('generator_capacity')
producer._capacity.set_const_VP(0)
producer.set_capacity(500)
producer._transfer = ValuedParamHandler('transfer_function')
producer._transfer.set_transfer_VP(transfer_function)

# test getters
# TODO this needs significant expanding, it was originally
# written to test things that are now deprecated.
if producer.get_tracking_vars() != ['production']:
  print('Error: tracking vars are incorrect')
  results['fail'] += 1
else:
  results['pass'] += 1

print(results)
sys.exit(results['fail'])

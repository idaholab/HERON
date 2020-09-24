'''
Test that meta does not get nested when producing with a transfer function
'''

import os
import sys
import xml.etree.ElementTree as ET

# Load HERON tools
HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,'src'))
sys.path.append(HERON_LOC)
import Components
import ValuedParams
import _utils as hutils
sys.path.pop()

# Load RAVEN tools
framework_path = hutils.get_raven_loc()
sys.path.append(framework_path)
from utils import InputData, xmlUtils,InputTypes
import MessageHandler
sys.path.pop()

# Set up the dummy transfer function
def transfer_function(method, requests, inputs):
    # Return the given request and the given meta (inputs)
    return {'electricity': 0}, inputs

# Set up the component
producer = Components.Producer()
producer.messageHandler = MessageHandler.MessageHandler()
producer.messageHandler.verbosity = 'debug'
# producer.messageHandler.callerLength = 0
# producer.messageHandler.tagLength = 0
producer._capacity_var = 'electricity'
producer._capacity = ValuedParams.ValuedParam('generator_capacity')
producer.set_capacity(500)
producer._transfer = ValuedParams.ValuedParam('transfer_function')
producer._transfer.type = 'Function'
producer._transfer._obj = ValuedParams.ValuedParam('transfer_function_obj')
producer._transfer._obj.evaluate = transfer_function

# Make the production request
request = {'electricity': 0}
meta = {'stuff': 'things'}
raven_vars = {}
dispatch = {}
t = 0
print('Meta before production:', meta)
stuff, meta = producer.produce(request, meta, raven_vars, dispatch, t)
print('Meta after production:', meta)


if 'meta' in meta:
    print('Error: meta came back nested.')
    sys.exit(1)
sys.exit(0)
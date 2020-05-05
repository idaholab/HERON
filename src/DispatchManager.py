"""
  Class for managing interactions with the Dispatchers.
"""

import os
import sys

# set up path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _utils as hutils
import SerializationManager

raven_path = hutils.get_raven_loc()

cashflow_path = hutils.get_cashflow_loc(raven_path=raven_path)

def run(raven, raven_dict):
  """
    API for external models.
    @ In, raven, object, RAVEN variables object
    @ In, raven_dict, dict, additional RAVEN information
  """
  path = os.path.join(os.getcwd(), '..', 'heron.lib') # TODO custom name?
  # load library file
  case, components, sources = SerializationManager.load_heron_lib(path, retry=6)
  # get appropriate dispatcher
  dispatcher = case.dispatcher
  dispatcher.initialize(raven_path=raven_path,
                        cashflow_path=cashflow_path
                       )
  # TODO clustering, multiyear, etc
  # TODO extract ARMA and other variables from RAVEN to pass through? Or just pass it all?
  dispatcher.dispatch(raven, raven_dict, case, components, sources)


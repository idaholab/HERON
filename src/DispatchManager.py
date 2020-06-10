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

class DispatchRunner:
  """
    Manages the interface between RAVEN and running the dispatch
  """
  # TODO move naming templates to a common place for consistency!
  naming_template = {'comp capacity': '{comp}_capacity',
                     'dispatch var': 'Dispatch__{comp}__{res}',
                    }

  def __init__(self):
    """
      Constructor. Note instances of this are tied to single ExternalModel runs.
      @ In, None
      @ Out, None
    """
    self._case = None              # HERON case
    self._components = None        # HERON components list
    self._sources = None           # HERON sources (placeholders) list

  def load_heron_lib(self, path):
    """
      Loads HERON objects from library file.
      @ In, path, str, path (including filename) to HERON library
      @ Out, None
    """
    case, components, sources = SerializationManager.load_heron_lib(path, retry=6)
    # arguments
    self._case = case              # HERON case
    self._components = components  # HERON components list
    self._sources = sources        # HERON sources (placeholders) list
    # derivative
    self._dispatcher = self._case.dispatcher

  def extract_variables(self, raven, raven_dict):
    """
      Extract variables from RAVEN and apply them to HERON objects
      @ In, raven, object, RAVEN external model object
      @ In, raven_dict, dict, RAVEN input dictionary
      @ Out, None
    """
    # TODO magic keywords (e.g. verbosity, MAX_TIMES, MAX_YEARS, ONLY_DISPATCH, etc)
    # component capacities
    for comp in self._components:
      name = self.naming_template['comp capacity'].format(comp=comp.name)
      update_capacity = raven_dict.get(name) # TODO is this ever not provided?
      comp.set_capacity(update_capacity)
    # TODO other case, component properties

  def run(self):
    """
      Runs dispatcher.
      @ In, None
      @ Out, None?
    """
    # TODO for each segment/cluster ..
    # TODO for each year ... ?
    ## TODO create "time" variable?
    dispatch = self._dispatcher.dispatch(self._case,
                                         self._components,
                                         self._sources,
                                         {})
    # TODO collect data per year/cluster/etc
    return dispatch

  def save_variables(self, raven, dispatch):
    """ generates RAVEN-acceptable variables TODO """
    # TODO clustering, multiyear
    # TODO should this be a Runner method or separate?
    template = self.naming_template['dispatch var']
    for comp_name, data in dispatch.items():
      for resource, usage in data.items():
        name = template.format(comp=comp_name, res=resource)
        setattr(raven, name, usage)
        # TODO indexMap?




def run(raven, raven_dict):
  """
    # TODO split into dispatch manager class and dispatch runner external model
    API for external models.
    This is run as part of the INNER ensemble model, run after the synthetic history generation
    @ In, raven, object, RAVEN variables object
    @ In, raven_dict, dict, additional RAVEN information
  """
  path = os.path.join(os.getcwd(), '..', 'heron.lib') # TODO custom name?
  # build runner
  runner = DispatchRunner()
  # load library file
  runner.load_heron_lib(path)
  # load data from RAVEN
  runner.extract_variables(raven, raven_dict)
  # TODO clustering, multiyear, etc?
  dispatch = runner.run()
  runner.save_variables(raven, dispatch)


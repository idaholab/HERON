
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Interface for user-provided dispatching strategies.
"""
import os
import inspect
import numpy as np

from ravenframework.utils import utils, InputData, InputTypes
from .Dispatcher import Dispatcher
from .DispatchState import NumpyState

class Custom(Dispatcher):
  """
    Base class for strategies for consecutive dispatching in a continuous period.
  """
  # ---------------------------------------------
  # INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('custom', ordered=False, baseNode=None)
    specs.addSub(InputData.parameterInputFactory('location', contentType=InputTypes.StringType,
        descr=r"""The hard drive location of the custom dispatcher. Relative paths are taken with
              respect to the HERON run location. Custom dispatchers must implement
              a \texttt{dispatch} method that accepts the HERON case, components, and sources; this
              method must return the activity for each resource of each component."""))
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    Dispatcher.__init__(self)
    self.name = 'CustomDispatcher'
    self._usr_loc = None # user-provided path to custom dispatcher module
    self._file = None    # resolved pathlib.Path to the custom dispatcher module

  def read_input(self, inputs):
    """
      Loads settings based on provided inputs
      @ In, inputs, InputData.InputSpecs, input specifications
      @ Out, None
    """
    usr_loc = inputs.findFirst('location')
    if usr_loc is None:
      raise RuntimeError('No <location> provided for <custom> dispatch strategy in <Case>!')
    # assure python extension, omitting it is a user convenience
    if not usr_loc.value.endswith('.py'):
      usr_loc.value += '.py'
    self._usr_loc = os.path.abspath(os.path.expanduser(usr_loc.value))

  def initialize(self, case, components, sources, **kwargs):
    """
      Initialize dispatcher properties.
      @ In, case, Case, HERON case instance
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    start_loc = case.run_dir
    file_loc = os.path.abspath(os.path.join(start_loc, self._usr_loc))
    # check that it exists
    if not os.path.isfile(file_loc):
      raise IOError(f'Custom dispatcher not found at "{file_loc}"! (input dir "{start_loc}", provided path "{self._usr_loc}"')
    self._file = file_loc
    print(f'Loading custom dispatch at "{self._file}"')
    # load user module
    load_string, _ = utils.identifyIfExternalModelExists(self, self._file, '')
    module = utils.importFromPath(load_string, True)
    # check it works as intended
    if not 'dispatch' in dir(module):
      raise IOError(f'Custom Dispatch at "{self._file}" does not have a "dispatch" method!')
    # TODO other checks?

  def dispatch(self, case, components, sources, meta):
    """
      Performs technoeconomic dispatch.
      @ In, case, Case, HERON case
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources
      @ Out, results, dict, economic and production metrics
    """
    # load up time indices
    t_start, t_end, t_num = self.get_time_discr()
    time = np.linspace(t_start, t_end, t_num) # Note we don't care about segment/cluster here
    # load user module
    load_string, _ = utils.identifyIfExternalModelExists(self, self._file, '')
    module = utils.importFromPath(load_string, True)
    state = NumpyState()
    indexer = meta['HERON']['resource_indexer']
    state.initialize(components, indexer, time)
    # run dispatch
    results = module.dispatch(meta, state)
    # TODO: Check to make sure user has uploaded all activity data.
    return state



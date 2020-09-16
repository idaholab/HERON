
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
import numpy as np
from io import StringIO

class DispatchState:
  """ utility that expresses the activity (i.e. production level) of all the components in the system """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self._components = None # list of HERON Component objects
    self._resources = None  # Map of resources to indices for components, as {comp.name: {res, r}}
    self._times = None      # numpy array of time values, monotonically increasing

  def initialize(self, components, resources_map, times):
    """
      Set up dispatch state to hold data
      @ In, components, list, HERON components to be stored
      @ In, resources_map, dict, map of resources to indices for each component
      @ In, times, list, float times to store activity
      @ Out, None
    """
    self._components = components
    self._resources = resources_map
    self._times = times

  def __repr__(self):
    """
      Compiles string representation of object.
      @ In, None
      @ Out, repr, str, string representation
    """
    return '<HERON generic DispatchState object>'

  def get_activity(self, comp, res, time, **kwargs):
    """
      Getter for activity level.
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, res, string, name of resource to retrieve
      @ In, time, float, time at which activity should be provided
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    r = self._resources[comp][res]
    t = np.searchsorted(self._times, time) # TODO protect against value not present
    return self.get_activity_indexed(comp, r, t, **kwargs)

  def set_activity(self, comp, res, time, value, **kwargs):
    """
      Setter for activity level.
      @ In, comp, HERON Component, component whose information should be set
      @ In, res, string, name of resource to retrieve
      @ In, time, float, time at which activity should be provided
      @ In, value, float, activity level; note positive is producting, negative is consuming
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, None
    """
    r = self._resources[comp][res]
    t = np.searchsorted(self._times, time) # TODO protect against value not present
    self.set_activity_indexed(comp, r, t, value, **kwargs)

  def get_activity_indexed(self, comp, r, t, **kwargs):
    """
      Getter for activity level, using indexes instead of values for r and t
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, r, int, index of resource to retrieve (as given by meta[HERON][resource_indexer])
      @ In, t, int, index of time at which activity should be provided
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    # to be overwritten by implementing classes
    raise NotImplementedError

  def set_activity_indexed(self, comp, r, t, value, **kwargs):
    """
      Getter for activity level, using indexes instead of values for r and t
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, r, int, index of resource to retrieve (as given by meta[HERON][resource_indexer])
      @ In, t, int, index of time at which activity should be provided
      @ In, value, float, value to set for activity
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    # to be overwritten by implementing classes
    raise NotImplementedError

  def create_raven_vars(self, template):
    """
      Writes out RAVEN variables as expected
      @ In, template, str, formating string for variable names (using {comp}, {res})
      @ Out, data, dict, map of raven var names to numpy array data
    """
    #template = 'Dispatch__{c}__{r}' # standardized via input
    data = {}
    for comp in self._components:
      for res, r in self._resources[comp].items():
        result = np.empty(len(self._times))
        for t, time in enumerate(self._times):
          result[t] = self.get_activity_indexed(comp, r, t)
        data[template.format(comp=comp.name, res=res)] = result
    return data

# NumpyState is the nominal DispatchState implementation
class NumpyState(DispatchState):
  """ implemenatation of DispatchState using Numpy. A good nominal choice if additional functionality isn't needed. """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    DispatchState.__init__(self)
    self._data = None # dict of numpy ND array for data

  def initialize(self, components, resources_map, times):
    """
      Set up dispatch state to hold data
      @ In, components, list, HERON components to be stored
      @ In, resources_map, dict, map of resources to indices for each component
      @ In, time, list, float times to store
      @ Out, None
    """
    DispatchState.initialize(self, components, resources_map, times)
    self._data = {}
    for comp in components:
      self._data[comp] = np.zeros((len(self._resources[comp]), len(times)))

  def __repr__(self):
    """
      Compiles string representation of object.
      @ In, None
      @ Out, repr, str, string representation
    """
    msg = StringIO()
    msg.write('<HERON NumpyState dispatch record: \n')
    for comp in self._data:
      resources = self._resources[comp]
      msg.write(f'   component: {comp.name}\n')
      for res, r in resources.items():
        msg.write(f'      {res}: {self._data[comp][r]}\n')
    msg.write('END NumpyState dispatch record>')
    return msg.getvalue()

  def get_activity_indexed(self, comp, r, t, **kwargs):
    """
      Getter for activity level.
      Note, if any of the arguments are "None" it is assumed that means "all"
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, r, int, index of resource to retrieve (as given by meta[HERON][resource_indexer])
      @ In, t, int, index of time at which activity should be provided
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    return self._data[comp][r, t]

  def set_activity_indexed(self, comp, r, t, value, **kwargs):
    """
      Setter for activity level.
      @ In, comp, HERON Component, component whose information should be set
      @ In, res, string, name of resource to retrieve
      @ In, time, float, time at which activity should be provided
      @ In, value, float, activity level; note positive is producting, negative is consuming
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, None
    """
    self._data[comp][r, t] = value

  def set_activity_vector(self, comp, res, start_time, end_time, values):
    """
      Shortcut utility for setting values all-at-once in a vector.
      @ In, comp, HERON Component, component whose information should be set
      @ In, res, string, name of resource to retrieve
      @ In, start_time, int, first time index at which activity is provided
      @ In, end_time, int, last time at which activity is provided (not inclusive)
      @ In, values, np.array, activity level; note positive is producting, negative is consuming
      @ Out, None
    """
    r = self._resources[comp][res]
    self._data[comp][r, start_time:end_time] = values

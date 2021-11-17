
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

  def get_activity(self, comp, activity, res, time, **kwargs):
    """
      Getter for activity level.
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, activity, str, tracking variable name for activity subset
      @ In, res, string, name of resource to retrieve
      @ In, time, float, time at which activity should be provided
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    r = self._resources[comp][res]
    t = np.searchsorted(self._times, time) # TODO protect against value not present
    return self.get_activity_indexed(comp, activity, r, t, **kwargs)

  def set_activity(self, comp, activity, res, time, value, **kwargs):
    """
      Setter for activity level.
      @ In, comp, HERON Component, component whose information should be set
      @ In, activity, str, tracking variable name for activity subset
      @ In, res, string, name of resource to retrieve
      @ In, time, float, time at which activity should be provided
      @ In, value, float, activity level; note positive is producting, negative is consuming
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, None
    """
    r = self._resources[comp][res]
    t = np.searchsorted(self._times, time) # TODO protect against value not present
    self.set_activity_indexed(comp, activity, r, t, value, **kwargs)

  def get_activity_indexed(self, *args, **kwargs):
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

  def set_activity_indexed(self, *args, **kwargs):
    """
      Getter for activity level, using indexes instead of values for r and t
      @ In, args, list, additional pass-through arguments
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
      for tracker in comp.get_tracking_vars():
        for res, r in self._resources[comp].items():
          result = np.empty(len(self._times))
          for t, time in enumerate(self._times):
            result[t] = self.get_activity_indexed(comp, tracker, r, t)
          data[template.format(comp=comp.name, tracker=tracker, res=res)] = result
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
      for tag in comp.get_tracking_vars():
        self._data[f'{comp.name}_{tag}'] = np.zeros((len(self._resources[comp]), len(times)))

  def __repr__(self):
    """
      Compiles string representation of object.
      @ In, None
      @ Out, repr, str, string representation
    """
    msg = StringIO()
    msg.write('<HERON NumpyState dispatch record: \n')
    for key, act_data in self._data.items():
      name, activity = key.split('_')
      # find corresponding component
      for comp, resources in self._resources.items():
        if comp.name == name:
          break
      resources = self._resources[comp]
      msg.write(f'   component: {name} activity: {activity}\n')
      for res, r in resources.items():
        msg.write(f'      {res}: {act_data[r]}\n')
    msg.write('END NumpyState dispatch record>')
    return msg.getvalue()

  def get_activity_indexed(self, comp, activity, r, t, **kwargs):
    """
      Getter for activity level.
      Note, if any of the arguments are "None" it is assumed that means "all"
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, activity, str, tracking variable name for activity subset
      @ In, r, int, index of resource to retrieve (as given by meta[HERON][resource_indexer])
      @ In, t, int, index of time at which activity should be provided
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    return self._data[f'{comp.name}_{activity}'][r, t]

  def set_activity_indexed(self, comp, activity, r, t, value, **kwargs):
    """
      Setter for activity level.
      @ In, comp, HERON Component, component whose information should be set
      @ In, activity, str, tracking variable name for activity subset
      @ In, res, string, name of resource to retrieve
      @ In, time, float, time at which activity should be provided
      @ In, value, float, activity level; note positive is producting, negative is consuming
      @ In, kwargs, dict, additional pass-through keyword arguments
      @ Out, None
    """
    self._data[f'{comp.name}_{activity}'][r, t] = value

  # def set_activity_vector(self, comp, tracker, res, start_time, end_time, values):
  def set_activity_vector(self, comp, res, values, tracker='production', start_idx=0, end_idx=None):
    """
      Shortcut utility for setting values all-at-once in a vector.
      @ In, comp, HERON Component, component whose information should be set
      @ In, res, string, name of resource to retrieve
      @ In, values, np.array, activity level; note positive is producting, negative is consuming
      @ In, tracker, str, optional, tracking variable name for activity subset, Default: 'production'
      @ In, start_idx, int, optional, first time index at which activity is provided, Default: 0
      @ In, end_idx, int, optional, last time index at which activity is provided, Default: None
      @ Out, None
    """
    if comp.get_interaction().is_type('Storage') and tracker == 'production':
      raise RuntimeError(f'Component "{comp.name}" is Storage and the provided tracker is "production"!')
    if end_idx is None:
      end_idx = len(self._times)

    r = self._resources[comp][res]
    self._data[f'{comp.name}_{tracker}'][r, start_idx:end_idx] = values

import numpy as np

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
      @ In, resources, list, string resources to be stored
      @ In, time, list, float times to store
      @ Out, None
    """
    self._components = components
    self._resources = resources_map
    self._times = times

  def get_activity(self, comp, res, time, **kwargs):
    """
      Getter for activity level.
      Note, if any of the arguments are "None" it is assumed that means "all"
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, res, string, name of resource to retrieve
      @ In, time, float, time at which activity should be provided (TODO should this be an int?)
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    r = self._resources[comp][res]
    t = np.searchsorted(self._times, time) # TODO protect against value not present
    return self.get_activity_indexed(comp, r, t, **kwargs)

  def set_activity(self, comp, res, time, value, **kwargs):
    """
      Setter for activity level.
      Note, if any of the arguments are "None" it is assumed that means "all"
      @ In, comp, HERON Component, component whose information should be set
      @ In, res, string, name of resource to retrieve
      @ In, time, float, time at which activity should be provided (TODO should this be an int?)
      @ In, value, float, activity level; note positive is producting, negative is consuming
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
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    raise NotImplementedError

  def set_activity_indexed(self, comp, r, t, value, **kwargs):
    """
      Getter for activity level, using indexes instead of values for r and t
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, r, int, index of resource to retrieve (as given by meta[HERON][resource_indexer])
      @ In, t, int, index of time at which activity should be provided
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    raise NotImplementedError




# NumpyState is the nominal DispatchState implementation
class NumpyState(DispatchState):
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
      @ In, resources, list, string resources to be stored
      @ In, time, list, float times to store
      @ Out, None
    """
    DispatchState.initialize(self, components, resources_map, times)
    self._data = {}
    for comp in components:
      self._data[comp] = np.zeros((len(self._resources[comp.name]), len(times)))

  def get_activity_indexed(self, comp, r, t):
    """
      Getter for activity level.
      Note, if any of the arguments are "None" it is assumed that means "all"
      @ In, comp, HERON Component, component whose information should be retrieved
      @ In, r, int, index of resource to retrieve (as given by meta[HERON][resource_indexer])
      @ In, t, int, index of time at which activity should be provided
      @ Out, activity, float, amount of resource "res" produced/consumed by "comp" at time "time";
                              note positive is producting, negative is consuming
    """
    return self._data[r, t]

  def set_activity_indexed(self, comp, res, time, value):
    """
      Setter for activity level.
      @ In, comp, HERON Component, component whose information should be set
      @ In, res, string, name of resource to retrieve
      @ In, time, float, time at which activity should be provided
      @ In, value, float, activity level; note positive is producting, negative is consuming
      @ Out, None
    """
    self._data[r, t] = value


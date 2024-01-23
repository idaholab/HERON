
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Base class for dispatchers.
"""
from ravenframework.utils import InputData
from ravenframework.BaseClasses import MessageUser, InputDataUser

class DispatchError(Exception):
    """
      Custom exception for dispatch errors.
    """
    pass


class Dispatcher(MessageUser, InputDataUser):
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
    specs = InputData.parameterInputFactory('Dispatcher', ordered=False, baseNode=None)
    # only place specifications that are true for ALL dispatchers here
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.name = 'BaseDispatcher'
    self._time_discretization = None # (start, end, num_steps) to build time discretization
    self._validator = None           # can be used to validate activity
    self._solver = None
    self._eps = 1e-9                 # small constant to add to denominator

  def read_input(self, inputs):
    """
      Loads settings based on provided inputs
      @ In, inputs, InputData.InputSpecs, input specifications
      @ Out, None
    """
    pass # add here if true for ALL dispatchers

  def initialize(self, case, components, sources, **kwargs):
    """
      Initialize dispatcher properties.
      @ In, case, Case, HERON case instance
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    pass

  # ---------------------------------------------
  # GETTER AND SETTERS
  def get_time_discr(self):
    """
      Retrieves the time discretization information.
      @ In, None
      @ Out, info, tuple, (start, end, number of steps) for time discretization
    """
    return self._time_discretization

  def set_time_discr(self, info):
    """
      Stores the time discretization information.
      @ In, info, tuple, (start, end, number of steps) for time discretization
      @ Out, None
    """
    assert info is not None
    # TODO is this the right idea?
    # don't expand into linspace right now, just store the pieces
    self._time_discretization = info

  def set_validator(self, validator):
    """
      Sets the dispatch validation instance to use in dispatching.
      @ In, validator, HERON Validator, instance of validator
      @ Out, None
    """
    self._validator = validator

  def get_solver(self):
    """
      Retrieves the solver information (if applicable)
      @ In, None
      @ Out, solver, str, name of solver used
    """
    return self._solver

  # ---------------------------------------------
  # API
  # TODO make this a virtual method?
  def dispatch(self, case, components, sources):
    """
      Performs technoeconomic dispatch.
      @ In, case, Case, HERON case
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources
      @ Out, results, dict, economic and production metrics
    """
    raise NotImplementedError # must be implemented by inheriting classes

  def validate(self, components, activity, times, meta):
    """
      Method to validate a dispatch activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ In, meta, dict, extra information needed for validation
      @ Out, validation, dict, information about validation
    """
    # default implementation
    if self._validator is not None:
      return self._validator.validate(components, activity, times, meta)
    else:
      # no validator, nothing needs to be changed
      return {}


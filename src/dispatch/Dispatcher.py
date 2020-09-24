
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Base class for dispatchers.
"""
from utils import InputData, InputTypes

class Dispatcher:
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
    self.name = 'BaseDispatcher'
    self._time_discretization = None # (start, end, num_steps) to build time discretization
    self._validator = None           # can be used to validate activity

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

  def validate(self, components, activity, times):
    """
      Method to validate a dispatch activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ Out, validation, dict, information about validation
    """
    # default implementation
    if self._validator is not None:
      return self._validator.validate(components, activity, times)
    else:
      # no validator, nothing needs to be changed
      return {}

  # ---------------------------------------------
  # UTILITY METHODS
  def _compute_cashflows(self, components, activity, times, meta, state_args=None):
    """
      Method to compute CashFlow evaluations given components and their activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ In, meta, dict, additional info to be passed through to functional evaluations
      @ In, state_args, dict, optional, additional arguments to pass while getting activity state
      @ Out, total, float, total cashflows for given components
    """
    if state_args is None:
      state_args = {}
    total = 0
    specific_meta = dict(meta) # TODO what level of copying do we need here?
    resource_indexer = meta['HERON']['resource_indexer']
    #print('DEBUGG computing cashflows!')
    for comp in components:
      #print(f'DEBUGG ... comp {comp.name}')
      specific_meta['HERON']['component'] = comp
      comp_subtotal = 0
      for t, time in enumerate(times):
        #print(f'DEBUGG ... ... time {t}')
        # NOTE care here to assure that pyomo-indexed variables work here too
        specific_activity = {}
        for resource in resource_indexer[comp]:
          specific_activity[resource] = activity.get_activity(comp, resource, time, **state_args)
        specific_meta['HERON']['time_index'] = t
        specific_meta['HERON']['time_value'] = time
        cfs = comp.get_state_cost(specific_activity, specific_meta)
        time_subtotal = sum(cfs.values())
        comp_subtotal += time_subtotal
      total += comp_subtotal
    return total



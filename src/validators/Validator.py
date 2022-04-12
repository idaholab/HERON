
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Base class for validators.
"""
from ravenframework.utils import InputData, InputTypes

class Validator:
  """
    Base class for strategies for validating dispatch decisions.
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
    specs = InputData.parameterInputFactory('Validator', ordered=False, baseNode=None)
    # only place specifications that are true for ALL validators here
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'BaseValidator'

  def read_input(self, inputs):
    """
      Loads settings based on provided inputs
      @ In, inputs, InputData.InputSpecs, input specifications
      @ Out, None
    """
    pass

  def initialize(self, case, components, sources, **kwargs):
    """
      Initialize validator properties.
      @ In, case, Case, HERON case instance
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    pass

  # ---------------------------------------------
  # GETTER AND SETTERS
  # TODO

  # ---------------------------------------------
  # API
  # TODO make this a virtual method?
  def validate(self, case, components, sources, dispatch, meta):
    """
      Performs technoeconomic dispatch.
      @ In, case, Case, HERON case
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources
      @ In, dispatch, HERON DispatchState, proposed activity
      @ In, meta, dict, extra information
      @ Out, results, list(dict), list of violations with info about each (see Validator base class)
    """
    # The return results should be a list of dictionaries
    # Each entry should include the following:
    # {msg: human-readable error message,
    #  limit: new limit to be imposed (as a float),
    #  limit_type: type of new limit (such as "upper" or "lower"; see below),
    #  component: HERON component object that violated conditions,
    #  resource: string name of resource for which violation was observed,
    #  time: float time at which violation was observed,
    #  time_index: integer index of time at which violation was observed,
    # }
    #
    # limit_type:
    # This will have to evolve with time. Hopefully we can keep this list updated.
    # - 'upper': signifies an upper production limit at that time step
    # - 'lower': signifies a lower production limit at that time step
    # in the future, I imagine we'll have lower_grad and upper_grad as well, who knows what else.
    raise NotImplementedError # must be implemented by inheriting classes

  # ---------------------------------------------
  # UTILITY METHODS
  # TODO

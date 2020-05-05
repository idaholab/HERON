"""
  Base class for dispatchers.
"""
import InputData, InputTypes

class Dispatcher:
  """
    Base class for strategies for consecutive dispatching in a continuous period.
  """

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

  def dispatch(self, case, components, sources):
    """
      Performs technoeconomic dispatch.
      @ In, case, Case, HERON case
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources
      @ Out, results, dict, economic and production metrics
    """
    raise NotImplementedError # must be implemented by inheriting classes
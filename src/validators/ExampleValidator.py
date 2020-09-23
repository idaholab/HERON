
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Example class for validators.
"""
import numpy as np

from utils import InputData, InputTypes

from .Validator import Validator

class Example(Validator):
  """
    Example class for validating dispatch decisions.
    Arbitrarily, uses percent based ramping limits
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
    specs = Validator.get_input_specs()
    # TODO left for convenience
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'BaseValidator'
    self._allowable = 0.2
    self._tolerance = 1e-10

  def read_input(self, inputs):
    """
      Loads settings based on provided inputs
      @ In, inputs, InputData.InputSpecs, input specifications
      @ Out, None
    """
    pass

  # ---------------------------------------------
  # API
  def validate(self, components, dispatch, times):
    """
      Method to validate a dispatch activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ Out, errs, list, information about validation failures
    """
    errs = [] # TODO best format for this?
    for comp, info in dispatch._resources.items():
      for res in info:
        for t, time in enumerate(times):
          current = dispatch.get_activity(comp, res, time)
          if t > 0:
            previous = dispatch.get_activity(comp, res, times[t-1])
            delta = current - previous
            sign = np.sign(delta)
            if abs(delta) - self._allowable > self._tolerance:
              errs.append({'msg': f'Exceeded ramp of {self._allowable} with {delta:1.8e}',
                           'limit': previous + (sign * self._allowable),
                           'limit_type': 'lower' if (sign < 0) else 'upper',
                           'component': comp,
                           'resource': res,
                           'time': time,
                           'time_index': t,
                          })

    return errs

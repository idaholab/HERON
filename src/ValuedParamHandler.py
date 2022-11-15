
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the base class ValuedParam entity.
"""
from __future__ import unicode_literals, print_function
import os
import sys

from HERON.src import _utils as hutils
from HERON.src.ValuedParams import Parametric
from HERON.src.ValuedParams import factory as VPFactory

try:
  import ravenframework
except ModuleNotFoundError:
  framework_path = hutils.get_raven_loc()
  sys.path.append(framework_path)
from ravenframework.utils import InputData, InputTypes
from ravenframework.BaseClasses import MessageUser

class ValuedParamHandler(MessageUser):
  """
    This class enables the identification of runtime-evaluated variables
    with a variety of sources (fixed values, parametric values, data histories, function
    evaluations, etc).
  """
  @classmethod
  def get_input_specs(cls, name, descr=""):
    """
      Template for parameters that can take a scalar, an ARMA history, or a function
      @ In, name, string, name for spec (tag)
      @ In, descr, string, base description for item (this will add to it)
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory(name,
        descr=descr + r"""This value can be taken from any \emph{one} of the sources described below.""")
    # VP sources
    for vp_type in VPFactory.knownTypes():
      spec.addSub(vp_type.get_input_specs())
    # addons
    spec.addSub(InputData.parameterInputFactory('multiplier', contentType=InputTypes.FloatType,
        descr=r"""Multiplies any value obtained by this parameter by the given value. \default{1}"""))
    # for when the result obtained needs to grow from year to year
    # TODO
    # growth = InputData.parameterInputFactory('growth', contentType=InputTypes.FloatType,
    #     descr=r"""if this node is given, the value will be adjusted from cycle to cycle by the provided amount.""")
    # growth_mode = InputTypes.makeEnumType('growthType', 'growthType', ['linear', 'exponential'])
    # growth.addParam('mode', param_type=growth_mode, required=True,
    #     descr=r"""determines whether the growth factor should be taken as linear or exponential (compounding).""")
    # spec.addSub(growth)
    return spec

  def __init__(self, name):
    """
      Constructor.
      @ In, name, str, name of this valued param
      @ Out, None
    """
    super().__init__()
    self.name = name         # member whom this ValuedParam provides values, e.g. Component.economics.alpha
    self._vp = None          # ValuedParam instance
    self._multiplier = None  # scalar multiplier for evaluation values
    self._growth_val = None  # used to grow the value year-by-year
    self._growth_mode = None # mode for growth (e.g. exponenetial, linear)

  def __repr__(self):
    """
      String representation of this Handler and its VP
      @ In, None
      @ Out, repr, str, string representation
    """
    msg = f'<ValuedParam name: "{self.name}" type: "{self.type}">'
    return msg

  @property
  def type(self):
    """
      Provides type of underlying vp
      @ In, None
      @ Out, type, str, string type (class name) of VP
    """
    return self._vp.type

  def read(self, comp_name: str, spec: InputData.ParameterInput, mode: str, alias_dict=None):
    """
      Used to read valued param from XML input
      @ In, comp_name, str, name of component that this valued param will be attached to; only used for print messages
      @ In, spec, InputData params, input specifications
      @ In, mode, type of simulation calculation
      @ In, alias_dict, dict, optional, aliases to use for variable naming
      @ Out, signal, list, signals needed to evaluate this ValuedParam at runtime
    """
    # aliases get used to convert variable names, notably for the cashflow's "capacity"
    if alias_dict is None:
      alias_dict = {}
    # instantiate the requested ValuedParam
    found = False
    knownVPs = VPFactory.knownTypes()
    for sub in spec.subparts:
      # ValuedParam
      if sub.getName() in knownVPs:
        # check against multiple source specifications
        if found:
          self.raiseAnError(IOError, 'Only one ValuedParam type can be used per node; received ' +
                            f'multiple for comp "{comp_name}" node <{spec.getName()}>!')
        self._vp = VPFactory.returnInstance(sub.getName())
        signal = self._vp.read(comp_name, sub, mode, alias_dict=alias_dict)
        found = True
      ## other addons
      # multiplier
      elif sub.getName() == 'multiplier':
        self._multiplier = sub.value
      # growth # TODO this doesn't appear to be implemented currently
      # elif sub.getName() == 'growth':
      #   self._growth_val = sub.value
      #   self._growth_mode = sub.parameterValues['mode']
    if not found:
      self.raiseAnError(IOError, f'Component "{comp_name}" node <{spec.getName()}> expected a ValuedParam ' +
                       f'to define its value source, but none was found! Options include: {knownVPs}')
    return signal

  def crosscheck(self, interaction):
    """
      Perform checks to make sure VP is set up correctly.
      @ In, interaction, Component.Interaction, HERON Interaction that "owns" this VP
      @ Out, None (error raised if not correct)
    """
    self._vp.crosscheck(interaction)

  def get_source(self):
    """
      Access the source type and name of the VP
      @ In, None
      @ Out, kind, str, identifier for the expected DataGenerator type (as a string)
      @ Out, source_name, identifier for the name of the associated DataGenerator
    """
    return self._vp.get_source()

  def get_value(self, **kwargs):
    """
      Get fixed values, if any.
      @ In, None
      @ Out, value, object, VP's reported fixed value (may be None)
    """
    return self._vp.get_value(**kwargs)

  def set_value(self, value):
    """
      Get fixed values, if any.
      @ In, value, object, VP's new fixed value (may be None)
      @ Out,
    """
    # should this be allowed if it's not parametric?
    assert isinstance(self._vp, Parametric)
    return self._vp.set_value(value)

  def get_coefficients(self):
    """
      Provide coefficients for linear transformation
      @ In, None
      @ Out, coeffs, dict, coefficients
    """
    assert self.type == 'Linear'
    return self._vp.get_coefficients()

  def is_parametric(self):
    """
      Tell if VP is parametric type
      @ In, None
      @ Out, is, bool, True if parametric
    """
    return isinstance(self._vp, Parametric)

  def set_const_VP(self, value):
    """
      Force the Handler to set a ValuedParam without doing reading.
      Mostly for testing.
      @ In, value, float, fixed value
      @ Out, None
    """
    self._vp = VPFactory.returnInstance('fixed_value')
    self._vp.set_value(value)

  def set_transfer_VP(self, func):
    """
      Force the Handler to set a transfer ValuedParam without doing reading.
      Mostly for testing.
      @ In, func, function, transfer function
      @ Out, None
    """
    from HERON.src.Placeholders import Function
    self._vp = VPFactory.returnInstance('Function')
    self._vp.evaluate = func

  def set_object(self, obj):
    """
      Provides a reference to the requested object for the VP.
      Requests are made through the return of "self.read".
      @ In, obj, isntance, evaluation target
      @ Out, None
    """
    self._vp.set_object(obj)

  def evaluate(self, *args, util_factor=False, **kwargs):
    """
      Evaluate the ValuedParam, wherever it gets its data from
      @ In, args, list, positional arguments for ValuedParam
      @ In, kwargs, dict, keyword arguements for ValuedParam
      @ Out, evaluate, object, stuff from ValuedParam evaluation
    """
    data, meta = self._vp.evaluate(*args, **kwargs)
    if self._multiplier is not None:
      for key in data:
        data[key] *= self._multiplier
    return data, meta


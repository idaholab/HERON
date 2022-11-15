
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the ValuedParam entity.
  These are objects that need to return values, but come from
  a wide variety of different sources and may not be valued until run time.
"""
import sys
from HERON.src import _utils as hutils
try:
  import ravenframework
except ModuleNotFoundError:
  framework_path = hutils.get_raven_loc()
  sys.path.append(framework_path)
from ravenframework.utils import InputData, InputTypes
from ravenframework.BaseClasses import MessageUser

# class for potentially dynamically-evaluated quantities
class ValuedParam(MessageUser):
  """
    These are objects that need to return values, but come from
    a wide variety of different sources and may not be valued until run time.
  """

  @classmethod
  def get_input_specs(cls, name):
    """
      Define inputs for this VP.
      @ In, name, string, name for spec (tag)
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory(name)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, name, str, name of this valued param
      @ Out, None
    """
    super().__init__()
    self.type = self.__class__.__name__ # class type, for easy checking
    self._comp = None        # component who uses this valued param
    self._source_kind = None # if taken from a source, this is the string name of the source
    self._source_name = None # name of source object (HERON tracking name, given by user)
    self._target_obj = None  # Placeholder object for this VP, if needed
    self._value = None       # None for most VP, for Parametric may be valued

  def __repr__(self) -> str:
    """
      Return Object Representation String
      @ In, None
      @ Out, None
    """
    return f"<HERON {self._source_kind}>"

  def read(self, comp_name, spec, mode, alias_dict=None):
    """
      Used to read valued param from XML input
      Also determines what "needs" from external entities exist for this entity; these are returned
      as a list so that the input handler can connect the requisite pieces.
      @ In, comp_name, str, name of component that this valued param will be attached to; only used for print messages
      @ In, spec, InputData params, input specifications
      @ In, mode, type of simulation calculation
      @ In, alias_dict, dict, optional, aliases to use for variable naming
      @ Out, needs, list, signals needed to evaluate this ValuedParam at runtime
    """
    # aliases get used to convert variable names, notably for the cashflow's "capacity"
    if alias_dict is None:
      alias_dict = {}
    # base implementation doesn't indicate any source/signal, so we return an empty list
    return []

  def crosscheck(self, interaction):
    """
      Allows for post-reading, post-crossref checking to make sure everything is in place.
      @ In, interaction, HERON.Component.Interaction, interaction that "owns" this VP
      @ Out, None
    """
    pass # optional, use in subclasses as needed

  def get_source(self):
    """
      Accessor for the source type and name of this valued param
      @ In, None
      @ Out, kind, str, identifier for the style of valued param
      @ Out, name, str, name of the source
    """
    return self._source_kind, self._source_name

  def get_fixed_value(self):
    """
      Get the value for this parametric source.
      @ In, None
      @ Out, value, None, value
    """
    return None

  def set_object(self, obj):
    """
      Set the evaluation target of this valued param (e.g., function, ARMA, etc).
      Actual object is specific to the valued param itself.
      @ In, obj, instance, evaluation target
      @ Out, None
    """
    self._target_obj = obj

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, value, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, meta, dict, dictionary of meta (possibly changed during evaluation)
    """
    self.raiseAnError(NotImplementedError, 'Overwrite the "evaluate" method in the ValuedParam strategy!')


# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the ValuedParam entity.
  These are objects that need to return values, but come from
  a wide variety of different sources.
"""
from .ValuedParam import ValuedParam, InputData, InputTypes

# class for values based on dispatch activity
class Activity(ValuedParam):
  """
    Represents a ValuedParam that takes values from dispatching activity
  """
  # these types represent values that do not need to be evaluated at run time, as they are determined.
  @classmethod
  def get_input_specs(cls):
    """
      Template for parameters that can take a scalar, an ARMA history, or a function
      @ In, None
      @ Out, spec, InputData, value-based spec
    """
    spec = InputData.parameterInputFactory('activity', contentType=InputTypes.StringType,
        descr=r"""indicates that this value will be taken from the dispatched activity of this component.
              The value of this node should be the name of a resource that this component
              either produces or consumes. Note the sign of the activity by default will be negative
              for consumed resources and positive for produced resources.""")
    spec.addParam('tracking', param_type=InputTypes.StringType, required=False,
        descr=r"""Some kinds of components have multiple tracking variables. This attribute specifies
               which tracking variable is desired. Options are only "production" for Production and Demand
               units, and [level, charge, discharge] for Storage units. Default is the first entry listed.""")
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._source_kind = 'activity'
    self._var_name = None # name of the variable within the synth hist
    self._resource = None # name of the resource whose activity should be used
    self._tracking_var = None # specific tracking variable for the component

  def read(self, comp_name, spec, mode, alias_dict=None):
    """
      Used to read valued param from XML input
      @ In, comp_name, str, name of component that this valued param will be attached to; only used for print messages
      @ In, spec, InputData params, input specifications
      @ In, mode, type of simulation calculation
      @ In, alias_dict, dict, optional, aliases to use for variable naming
      @ Out, needs, list, signals needed to evaluate this ValuedParam at runtime
    """
    super().read(comp_name, spec, mode, alias_dict=None)
    subvar = spec.parameterValues.get('tracking', None)
    self._tracking_var = subvar # NOTE this gets fixed up in the crosscheck
    # aliases get used to convert variable names, notably for the cashflow's "capacity"
    if alias_dict is None:
      alias_dict = {}
    self._resource = spec.value
    # FIXME how to confirm this component actually produces/consumes/stores this resource??
    return []

  def crosscheck(self, interaction):
    """
      Allows for post-reading, post-crossref checking to make sure everything is in place.
      @ In, interaction, HERON.Component.Interaction, interaction that "owns" this VP
      @ Out, None
    """
    # use this chance to link the interaction tracking vars
    ok_trackers = interaction.get_tracking_vars()
    if self._tracking_var is None:
      self._tracking_var = ok_trackers[0]
      self.raiseAMessage(f'Tracking variable not specified; using "{self._tracking_var}" ...')
    else:
      if self._tracking_var not in ok_trackers:
        self.raiseAnError(f'Tracking variable "{self._tracking_var}" is not one of the variables ' +
                           f'tracked by this interaction! Options are: {ok_trackers}.')
    # check that the requested resource is actually used by this interaction
    available = interaction.get_resources()
    if self._resource not in available:
      str_avail = ['"{}"'.format(a) for a in available]
      self.raiseAnError(IOError, f'Requested <activity> value from resource "{self._resource}" but "{self._resource}" ' +
      f'was not found among this Component\'s input/output resources; options are:' +
      f'{", ".join(str_avail)}')

  def evaluate(self, inputs, target_var=None, aliases=None):
    """
      Evaluate this ValuedParam, wherever it gets its data from
      @ In, inputs, dict, stuff from RAVEN, particularly including the keys 'meta' and 'raven_vars'
      @ In, target_var, str, optional, requested outgoing variable name if not None
      @ In, aliases, dict, optional, alternate variable names for searching in variables
      @ Out, value, dict, dictionary of resulting evaluation as {vars: vals}
      @ Out, meta, dict, dictionary of meta (possibly changed during evaluation)
    """
    if aliases is None:
      aliases = {}
    # set the outgoing name for the evaluation results
    key = self._var_name if not target_var else target_var
    try:
      value = inputs['HERON']['activity'][self._tracking_var][self._resource]
    except KeyError as e:
      self.raiseAnError(RuntimeError, f'Resource "{self._resource}" was not found among those produced and ' +
                        'consumed by this component!')
    return {key: value}, inputs

# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the Component entity.
"""
from numbers import Real

from HERON.src.ValuedParams import factory as vp_factory
from HERON.src.ValuedParamHandler import ValuedParamHandler

from DOVE.src.Components import Component as DoveComponent
from DOVE.src.Interactions import Interaction as DoveInteraction
from DOVE.src.Interactions import Producer as DoveProducer
from DOVE.src.Interactions import Demand as DoveDemand
from DOVE.src.Interactions import Storage as DoveStorage
from DOVE.src.Economics import CashFlowGroup as DoveCashFlowGroup
from DOVE.src.Economics import CashFlow as DoveCashFlow

from ravenframework.utils import InputData
from ravenframework.utils.InputData import ParameterInput


class HeronComponent(DoveComponent):
  """
  Represents a unit in the grid analysis. 
  Each component has a single "interaction" that describes what it can do:
    -produce
    -store
    -demand
  """
  tag = "component"
  def __repr__(self) -> str:
    """
    String representation.
    @ In, None
    @ Out, __repr__, string representation
    """
    return f'<HERON Component "{self.name}">'

  @classmethod
  def get_input_specs(cls) -> type[ParameterInput]:
    """
    Collects input specifications for this class.
    @ In, None
    @ Out, input_specs, InputData, specs
    """
    ## DEVELOPER NOTE:
    ## You should NOT add new subspecs to this method unless they have nothing to
    ## do with DOVE (In which case maybe rethink its applicability to HERON).
    ## All InputSpecs for Interactions are defined within the DOVE.src.Interactions.
    ## If a new VP node needs to be added, find someway to add a fixed-value version
    ## to DOVE first! Then modify it in here. This will keep feature parity with DOVE.
    ## YOU HAVE BEEN WARNED!

    # Grab all the DOVE input specs -- these input specs have no ValuedParams in
    # them, so we need to modify the input spec to allow for those VPs
    input_specs = super().get_input_specs()

    # Define the subs to modify along with their configurations -- if we need to
    # modify more subs later, just add them to this dict and they'll be added.
    interact_subs_to_modify = {
      "capacity": {
        "add_params": [("resource", "resource")],
        "allowed": None # Meaning all "allowed" ValuedParams
      },
      "capacity_factor": {
        "add_params": [],
        "allowed": ['ARMA', 'CSV']
      },
      "minimum": {
        "add_params": [("resource", "resource")],
        "allowed": None,
      },
      "initial_stored": {
        "add_params": [],
        "allowed": None,
      },
      "strategy": {
        "add_params": [],
        "allowed": ['Function'],
      },
    }

    econ_subs_to_modify = {
      "driver":{
        "add_params": [],
        "allowed": ['activity', 'variable', 'Function'],
      },
      "reference_price": {
        "add_params": [],
        "allowed": None,
      },
      "reference_driver": {
        "add_params": [],
        "allowed": None,
      },
      "scaling_factor_x": {
        "add_params": [],
        "allowed": None,
      },
    }

    # Iterate over the subs to modify
    for sub in input_specs.subs:
      for sub_name, config in interact_subs_to_modify.items():
        current_sub = sub.getSub(sub_name)
        if current_sub is not None:
          new_sub = vp_factory.make_input_specs(sub_name, descr=sub.description, allowed=config["allowed"])
          # Add parameters if any
          for param_name, param_key in config["add_params"]:
            new_sub.addParam(param_name, descr=current_sub.parameters[param_key]['description'])
            # Replace the old sub with the new one
          _ = sub.popSub(sub_name)
          sub.addSub(new_sub)

    # We have to iterate a level deeper to get to CashFlow nodes
    for sub in input_specs.subs:
      if sub.getName() == "economics":
        for econ_sub in sub.subs:
          if econ_sub.getName() == "CashFlow":
            for sub_name, config in econ_subs_to_modify.items():
              current_sub = econ_sub.getSub(sub_name)
              if current_sub is not None:
                new_sub = vp_factory.make_input_specs(sub_name, descr=sub.description, allowed=config["allowed"])
                # Add parameters if any
                for param_name, param_key in config["add_params"]:
                  new_sub.addParam(param_name, descr=current_sub.parameters[param_key]['description'])
                # FIXME: This is bad and I should be punished...
                # Since <levelized_cost> is a child node of <reference_price> we need to re-add it to the input spec
                # when recreating the ValuedParam version of <reference_price>. Otherwise it gets deleted and is no
                # longer seen as an available input option. We should find a way to dynamically add child subs by perhaps
                # adding a key to `econ_subs_to_modify` dictionary like "add_subs". For now we add a conditional checking
                # if we are modifying <reference_price> and then directly add <levelized_cost> to the new input definition.
                if sub_name == "reference_price":
                  levelized_cost = InputData.parameterInputFactory(
                    "levelized_cost", strictMode=True, descr="indicates to solve for levelized price related to the cashflow"
                  )
                  new_sub.addSub(levelized_cost)
                # Replace the old sub with the new one
                _ = econ_sub.popSub(sub_name)
                econ_sub.addSub(new_sub)

    return input_specs

  def read_input(self, xml) -> None:
    """
    Sets settings from input file
    @In, xml, xml.etree.ElementTree.Element, input from user
    @In, mode, string, case mode to operate in (e.g. 'sweep' or 'opt')
    @Out, None
    """
    # get specs for allowable inputs
    specs = self.get_input_specs()()
    specs.parseNode(xml)
    self.name = specs.parameterValues['name']
    interaction_map = {
      "produces": HeronProducer,
      "stores": HeronStorage,
      "demands": HeronDemand
    }

    found_interactions: dict
    not_found_in_spec: list
    found_interactions, not_found_in_spec = specs.findNodesAndExtractValues(interaction_map.keys())
    if all((interaction == 'no-default' for interaction in found_interactions.values())):
      self.raiseAnError(NotImplementedError, f"No interaction found for Component '{self.name}'")
    elif len(not_found_in_spec) < 2:
      self.raiseAnError(NotImplementedError, f"A Component can only have one interaction! Check Component '{self.name}'")

    for item in specs.subparts:
      item_name = item.getName()
      if item_name in interaction_map:
        interaction_instance = interaction_map[item_name](messageHandler=self.messageHandler)
        interaction_instance.read_input(item, self.name)
        self._interaction = interaction_instance
      elif item_name == 'economics':
        cashflows = HeronCashFlowGroup(self, messageHandler=self.messageHandler)
        cashflows.read_input(item)
        self._economics = cashflows

  def get_capacity(self, meta, raw=False):
    """
    returns the capacity of the interaction of this component
    @In, meta, dict, arbitrary metadata from HERON
    @In, raw, bool, optional, if True then return the ValuedParam instance for capacity, instead of the evaluation
    @Out, capacity, float (or ValuedParam), the capacity of this component's interaction
    """
    return self._interaction.get_capacity(meta, raw=raw)

  def get_uncertain_cashflow_params(self):
    """
    Get all uncertain economic parameters
    @In, None
    @Out, params, dict, the uncertain parameters
    """
    params = {}
    for cf in self.get_cashflows():
      uncertain = cf.get_uncertain_params()
      params |= {f"{self.name}_{k}": v for k, v in uncertain.items()}
    return params

class HeronCashFlowGroup(DoveCashFlowGroup):
  """
  Masks specific functionality from DoveCashFlowGroup to allow for ValuedParams.
  """
  def read_input(self, specs: ParameterInput) -> None:
    """
    Sets settings from input file
    @In, source, InputData.ParameterInput, input from user
    @Out, None
    """
    for item in specs.subparts:
      item_name = item.getName()
      if item_name == "lifetime":
        self._lifetime = item.value
      elif item_name == "CashFlow":
        cashflow = HeronCashFlow(self._component)
        cashflow.read_input(item)
        self._cash_flows.append(cashflow)

class HeronCashFlow(DoveCashFlow):
  """
  Masks specific functionality from DoveCashFlow to allow for ValuedParams.
  """
  def __repr__(self) -> str:
    """
    String representation.
    @In, None
    @Out, __repr__, string representation
    """
    return f'<HERON CashFlow "{self.name}">'

  def _set_value(self, name: str, spec: ParameterInput) -> None:
    """
    Utilitly method to set ValuedParam members via reading input specifications.
    @In, name, str, member variable name (e.g. self.<name>)
    @In, spec, InputData params, input parameters
    @Out, None
    """
    vp = ValuedParamHandler(name)
    signal = vp.read(f'CashFlow \'{self.name}\'', spec)
    self._signals.update(signal)
    self._crossrefs[name] = vp
    # standard alias: redirect "capacity" variable
    if isinstance(vp, vp_factory.returnClass('variable')) and vp.get_raven_var() == 'capacity':
      #NOTE: we are assuming here that capacity_factors are only applied in dispatch and
      # are not a variable in the outer optimization.
      vp = self._component.get_capacity_param()
    setattr(self, name, vp)

  def _set_fixed_param(self, name: str, value: Real) -> None:
    """
    Fixes a ValuedParam to have a constant value
    @In, name, str, name of member to store on "self"
    @In, value, float, value to set for ValuedParam
    @Out, None
    """
    vp = ValuedParamHandler(name)
    vp.set_const_VP(value)
    setattr(self, name, vp)

  def get_uncertain_params(self) -> dict[str, ValuedParamHandler]:
    """
    Return all cashflow parameters that are random variables.
    @In, None
    @Out, uncertain_params, dict[str, RandomVariable], the uncertain cashflow parameters
    """
    params = ["_driver", "_alpha", "_reference", "_scale"]
    uncertain_params = {}
    for param_name in params:
      if (param := getattr(self, param_name)).type == 'RandomVariable':
        uncertain_params[param_name[1:]] = param
    return uncertain_params

class HeronInteraction(DoveInteraction):
  """
  Base class for component interactions (e.g. Producer, Storage, Demand)
  """
  tag = 'interacts'

  def _set_fixed_value(self, name: str, value: Real) -> None:
    """
    Sets a class attribute to a constant ValuedParam that will be evaluated at Runtime.
    This is a masked DoveInteraction method. The original method sets a literal value.
    @In, name, str, name of class attribute to create
    @In, value, Number, literal to set 
    @Out, None
    """
    vp = ValuedParamHandler(name)
    vp.set_const_VP(value)
    setattr(self, name, vp)

  def _set_value(self, name: str, comp_name: str, spec: ParameterInput) -> None:
    """
    Sets up use of a ValuedParam for this interaction for the "name" attribute of this class.
    @In, name, str, name of member of this class
    @In, comp_name, str, name of associated component
    @In, spec, InputParam, input specifications
    @Out, None
    """
    vp = ValuedParamHandler(name)
    signal = vp.read(comp_name, spec)
    self._signals.update(signal)
    self._crossrefs[name] = vp
    setattr(self, name, vp)

  def get_capacity(self, meta, raw=False):
    """
    Returns an evaluated value unless "raw" is True, then gives ValuedParam
    @ In, meta, dict, additional variables to pass through
    @ In, raw, bool, optional, if True then provide ValuedParam instead of evaluation
    @ Out, evaluated, float or ValuedParam, requested value
    @ Out, meta, dict, additional variable passthrough
    """
    if raw:
      #NOTE: not returing capacity_factor since it will not be used as a variable
      return self._capacity
    meta['request'] = {self._capacity_var: None}
    evaluated, meta = self._capacity.evaluate(meta, target_var=self._capacity_var)
    # apply capacity factor to get actual capacity for given timestep
    if self._capacity_factor is not None:
      capacity_factor = self._capacity_factor.evaluate(meta, target_var=self._capacity_var)[0]
      evaluated[self._capacity_var] *= capacity_factor[self._capacity_var]
    return evaluated, meta

  def get_minimum(self, meta, raw=False):
    """
    Returns an evaluated value unless "raw" is True, then gives ValuedParam
    @ In, meta, dict, additional variables to pass through
    @ In, raw, bool, optional, if True then provide ValuedParam instead of evaluation
    @ Out, evaluated, float or ValuedParam, requested value
    @ Out, meta, dict, additional variable passthrough
    """
    if raw:
      return self._minimum
    cap_var = self.get_capacity_var()
    if self._minimum is None:
      evaluated = {cap_var: 0.0}
    else:
      meta['request'] = {cap_var: None}
      evaluated, meta = self._minimum.evaluate(meta, target_var=cap_var)
      # check that min value is acceptable [0,1]
      # TODO it would be better to be able to check this before run-time, but we don't have a method
      #   in place to check e.g. ARMA,
      value = evaluated[cap_var]
      if not (0 <= value <= 1):
        self.raiseAnError(ValueError, f'While calculating minimum operating level for component "{self.tag}", ' +
            f'an invalid percent was provided/calculated ({value}). Minimums should be between 0 and 1, inclusive.')
      # convert percentage to real value
      evaluated[cap_var] = self.get_capacity(meta)[0][cap_var] * value
    return evaluated, meta

class HeronProducer(HeronInteraction, DoveProducer):
  """
  Explains a particular interaction, where resources are consumed to produce other resources
  """
  pass

class HeronStorage(HeronInteraction, DoveStorage):
  """
  Explains a particular interaction, where a resource is stored and released later
  """
  pass

class HeronDemand(HeronInteraction, DoveDemand):
  """
  Explains a particular interaction, where a resource is demanded
  """
  pass

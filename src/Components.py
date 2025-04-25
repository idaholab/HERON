# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the Component entity.
"""
from typing import Union, cast
from collections import defaultdict
from HERON.src.ValuedParams import factory as vp_factory
from HERON.src.ValuedParamHandler import ValuedParamHandler
from HERON.src.Placeholders import Placeholder

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
    @ In, xml, xml.etree.ElementTree.Element, input from user
    @ Out, None
    """
    specs = self.get_input_specs()()
    specs.parseNode(xml)
    # We need to overwrite DoveComponent.read_input() so we can
    # substitute our special HeronInteraction types that can handle ValuedParams!
    interaction_map = {"produces": HeronProducer, "stores": HeronStorage, "demands": HeronDemand}
    self.assign_attrs_from_specs(specs, interaction_map, HeronCashFlowGroup)

  def get_crossrefs(self) -> dict[Union['HeronInteraction', 'HeronCashFlow'], defaultdict[str, ValuedParamHandler]]:
    """
    Collect the required value entities needed for this component to function.
    @ In, None
    @ Out, crossrefs, dict, mapping of dictionaries with information about the entities required.
    """
    crossrefs: dict[Union[HeronInteraction,HeronCashFlow], defaultdict[str, ValuedParamHandler]] = {cast(HeronInteraction, self.interaction): cast(HeronInteraction, self.interaction).get_crossrefs()}
    crossrefs |= cast(HeronCashFlowGroup, self.economics).get_crossrefs()
    return crossrefs

  def set_crossrefs(self, refs: dict[Union['HeronInteraction', 'HeronCashFlow'], defaultdict[str, Placeholder]]) -> None:
    """
    Connect cross-reference material from other entities to the ValuedParams in this component.
    @ In, refs, dict, dictionary of entity information
    @ Out, None
    """
    current_interaction = cast(HeronInteraction, self.interaction)
    for found_interaction in list(refs.keys()):
      # find associated interaction
      if current_interaction == found_interaction:
        current_interaction.set_crossrefs(refs.pop(found_interaction))
        break
    # send what's left to the economics
    cast(HeronCashFlowGroup, self.economics).set_crossrefs(refs)
    # if anything left, there's an issue
    assert not refs

  def get_uncertain_cashflow_params(self) -> dict[str, ValuedParamHandler]:
    """
    Get all uncertain economic parameters
    @ In, None
    @ Out, params, dict, the uncertain parameters
    """
    params: dict[str, ValuedParamHandler] = {}
    for cf in cast(list[HeronCashFlow], self.economics.cashflows):
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
    @ In, source, InputData.ParameterInput, input from user
    @ Out, None
    """
    for item in specs.subparts:
      match item.getName():
        case "lifetime":
          self.lifetime = item.value
        case "CashFlow":
          cashflow = HeronCashFlow(self._component)
          cashflow.read_input(item)
          self.cashflows.append(cashflow)

    if self.lifetime is None:
      self.raiseAnError(IOError, f'Component "{self.name}" is missing <lifetime> node!')

  def evaluate_cfs(self, activity, meta, marginal=False):
    """
    Calculates the incremental cost of a particular system configuration.
    @ In, activity, XArray.DataArray, array of driver-centric variable values
    @ In, meta, dict, additional user-defined meta
    @ In, marginal, bool, optional, if True then only get marginal cashflows (e.g. recurring hourly)
    @ Out, cost, dict, cash flow evaluations
    """
    # combine all cash flows into single cash flow evaluation
    if marginal:
      # FIXME assuming 'year' is the only non-marginal value
      # FIXME why is it "repeating" and not "Recurring"?
      cost = dict(
        (cf.name, cf.evaluate_cost(activity, meta))
        for cf in self.cashflows
        if (cf.type == "repeating" and cf.period != "year")
      )
    else:
      cost = dict((cf.name, cf.evaluate_cost(activity, meta)) for cf in self.cashflows)
    return cost

  def get_crossrefs(self) -> dict['HeronCashFlow', defaultdict[str, ValuedParamHandler]]:
    """
    Provides a dictionary of the entities needed by this cashflow group to be evaluated
    @ In, None
    @ Out, crossrefs, dict, dictionary of crossreferences needed (see ValuedParams)
    """
    crossrefs = dict((cast(HeronCashFlow, cf), cast(HeronCashFlow, cf).get_crossrefs()) for cf in self.cashflows)
    return crossrefs

  def set_crossrefs(self, refs: dict[Union['HeronInteraction','HeronCashFlow'], defaultdict[str, Placeholder]]) -> None:
    """
    Provides links to entities needed to evaluate this cash flow group.
    @ In, refs, dict, reference entities
    @ Out, None
    """
    # set up pointers
    for cf in list(refs.keys()):
      for try_match in self.cashflows:
        if try_match == cf:
          cast(HeronCashFlow, try_match).set_crossrefs(refs.pop(cast(HeronCashFlow, try_match)))
          break
      else:
        cf.set_crossrefs({}) #type: ignore



class HeronCashFlow(DoveCashFlow):
  """
  Masks specific functionality from DoveCashFlow to allow for ValuedParams.
  """
  def __repr__(self) -> str:
    """
    String representation.
    @ In, None
    @ Out, __repr__, string representation
    """
    return f'<HERON CashFlow "{self.name}">'

  def _set_value(self, name: str, spec: ParameterInput) -> None:
    """
    Utilitly method to set ValuedParam members via reading input specifications.
    @ In, name, str, member variable name (e.g. self.<name>)
    @ In, spec, InputData params, input parameters
    @ Out, None
    """
    vp = ValuedParamHandler(name)
    signal = vp.read(f'CashFlow \'{self.name}\'', spec)
    self._signals.update(signal)
    self._crossrefs[name] = vp
    setattr(self, name, vp)

  def _set_fixed_param(self, name: str, value: float) -> None:
    """
    Fixes a ValuedParam to have a constant value
    @ In, name, str, name of member to store on "self"
    @ In, value, float, value to set for ValuedParam
    @ Out, None
    """
    vp = ValuedParamHandler(name)
    vp.set_const_VP(value)
    setattr(self, name, vp)

  def get_crossrefs(self) -> defaultdict[str, ValuedParamHandler]:
    """
    Accessor for cross-referenced entities needed by this cashflow.
    @ In, None
    @ Out, crossrefs, dict, cross-referenced requirements dictionary
    """
    return self._crossrefs

  def set_crossrefs(self, refs: defaultdict[str, Placeholder]) -> None:
    """
    Setter for cross-referenced entities needed by this cashflow.
    @ In, refs, dict, cross referenced entities
    @ Out, None
    """
    # set up pointers
    for attr, obj in refs.items():
      valued_param = cast(ValuedParamHandler, self._crossrefs[attr])
      valued_param.set_object(obj)
    # check on VP setup
    for vp in self._crossrefs.values():
      cast(ValuedParamHandler, vp).crosscheck(self.component.interaction)

  def evaluate_cost(self, activity, values_dict):
    """
    Evaluates cost of a particular scenario provided by "activity".
    @ In, activity, pandas.Series, multi-indexed array of scenario activities
    @ In, values_dict, dict, additional values that may be needed to evaluate cost
    @ In, t, int, time index at which cost should be evaluated
    @ Out, cost, float, cost of activity
    """
    # note this method gets called a LOT, so speedups here are quite effective
    # add the activity to the dictionary
    values_dict["HERON"]["activity"] = activity
    params = self.calculate_params(values_dict)
    return params["cost"]

  def calculate_params(self, values_dict):
    """
    Calculates the value of the cash flow parameters.
    @ In, values_dict, dict, mapping from simulation variable names to their values (as floats or numpy arrays)
    @ Out, params, dict, dictionary of parameters mapped to values including the cost
    """
    # TODO maybe don't cast these as floats, as they could be symbolic expressions (seems unlikely)
    Dp = float(self._reference_driver.evaluate(values_dict, target_var="reference_driver")[0]["reference_driver"])
    x = float(self._scaling_factor_x.evaluate(values_dict, target_var="scaling_factor_x")[0]["scaling_factor_x"])
    a = self._alpha.evaluate(values_dict, target_var="reference_price")[0]["reference_price"]
    D = self._driver.evaluate(values_dict, target_var="driver")[0]["driver"]
    cost = a * (D / Dp) ** x
    params = {
      "alpha": a,
      "driver": D,
      "ref_driver": Dp,
      "scaling": x,
      "cost": cost,
    }  # TODO float(cost) except in pyomo it's not a float
    return params

  def get_uncertain_params(self) -> dict[str, ValuedParamHandler]:
    """
    Return all cashflow parameters that are random variables.
    @ In, None
    @ Out, uncertain_params, dict[str, RandomVariable], the uncertain cashflow parameters
    """
    params = ["_driver", "_alpha", "_reference_driver", "_scaling_factor_x"]
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

  def _set_fixed_value(self, name: str, value: float) -> None:
    """
    Sets a class attribute to a constant ValuedParam that will be evaluated at Runtime.
    This is a masked DoveInteraction method. The original method sets a literal value.
    @ In, name, str, name of class attribute to create
    @ In, value, Number, literal to set
    @ Out, None
    """
    vp = ValuedParamHandler(name)
    vp.set_const_VP(value)
    setattr(self, name, vp)

  def _set_value(self, name: str, comp_name: str, spec: ParameterInput) -> None:
    """
    Sets up use of a ValuedParam for this interaction for the "name" attribute of this class.
    @ In, name, str, name of member of this class
    @ In, comp_name, str, name of associated component
    @ In, spec, InputParam, input specifications
    @ Out, None
    """
    vp = ValuedParamHandler(name)
    signal = vp.read(comp_name, spec)
    self._signals.update(signal)
    self._crossrefs[name] = vp
    setattr(self, name, vp)

  def get_crossrefs(self) -> defaultdict[str, ValuedParamHandler]:
    """
    Getter.
    @ In, None
    @ Out, crossrefs, dict, resource references
    """
    return cast(defaultdict[str, ValuedParamHandler], self._crossrefs)

  def set_crossrefs(self, refs: defaultdict[str, Placeholder]) -> None:
    """
    Setter.
    @ In, refs, dict, resource cross-reference objects
    @ Out, None
    """
    # connect references to ValuedParams (Placeholder objects)
    for attr, obj in refs.items():
      valued_param = cast(ValuedParamHandler, self._crossrefs[attr])
      valued_param.set_object(obj)
    # perform crosscheck that VPs have what they need
    for vp in self._crossrefs.values():
      cast(ValuedParamHandler, vp).crosscheck(self)


  def set_capacity(self, cap) -> None:
    """
    Allows hard-setting the capacity of this interaction.
    This destroys any underlying ValuedParam that was there before.
    @ In, cap, float, capacity value
    @ Out, None
    """
    self._capacity.set_value(float(cap))

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
    meta['request'] = {self.capacity_var: None}
    evaluated, meta = self._capacity.evaluate(meta, target_var=self.capacity_var)
    # apply capacity factor to get actual capacity for given timestep
    if self._capacity_factor is not None:
      capacity_factor = self._capacity_factor.evaluate(meta, target_var=self.capacity_var)[0]
      evaluated[self.capacity_var] *= capacity_factor[self.capacity_var]
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
    cap_var = self.capacity_var
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
  def get_initial_level(self, meta):
    """
    Find initial level of the storage
    @ In, meta, dict, additional variable passthrough
    @ Out, initial, float, initial level
    """
    res = self.get_stored_resource()
    request = {res: None}
    meta["request"] = request
    pct = self._initial_stored.evaluate(meta, target_var=res)[0][res]
    if not (0 <= pct <= 1):
      self.raiseAnError(
        ValueError,
        f'While calculating initial storage level for storage "{self.tag}", '
        + f"an invalid percent was provided/calculated ({pct}). Initial levels should be between 0 and 1, inclusive.",
      )
    amt = pct * self.get_capacity(meta)[0][res]
    return amt


class HeronDemand(HeronInteraction, DoveDemand):
  """
  Explains a particular interaction, where a resource is demanded
  """
  pass


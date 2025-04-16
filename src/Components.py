
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the Component entity.
"""
import sys
from collections import defaultdict
import numpy as np
from HERON.src.base import Base
import xml.etree.ElementTree as ET
#from HERON.src.Economics import CashFlowUser
from HERON.src.ValuedParams import factory as vp_factory

from HERON.src.ValuedParamHandler import ValuedParamHandler
from HERON.src import _utils as hutils

from DOVE.src.Components import Component as DoveComponent
from DOVE.src.TransferFuncs import factory as tf_factory
from DOVE.src.Interactions import (Interaction as DoveInteraction,
                                   Producer as DoveProducer,
                                   Demand as DoveDemand,
                                   Storage as DoveStorage)
from DOVE.src.Economics import (CashFlowGroup as DoveCashFlowGroup,
                                CashFlow as DoveCashFlow)

try:
  import ravenframework
except ModuleNotFoundError:
  framework_path = hutils.get_raven_loc()
  sys.path.append(framework_path)
from ravenframework.utils import InputData, xmlUtils, InputTypes


class HeronComponent(DoveComponent):
  """
    Represents a unit in the grid analysis. Each component has a single "interaction" that
    describes what it can do (produce, store, demand)
  """
  tag = "component"
  def __repr__(self):
    """
    String representation.
    @ In, None
    @ Out, __repr__, string representation
    """
    return f'<HERON Component "{self.name}">'

  @classmethod
  def get_input_specs(cls):
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

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, kwargs, dict, optional, arguments to pass to other constructors
      @ Out, None
    """
    super().__init__(**kwargs)
    # Base.__init__(self, **kwargs)
    # HeronCashFlowGroup.__init__(self)
    self.name = None
    self._produces = []
    self._stores = []
    self._demands = []
    self.levelized_meta = {}


  def read_input(self, xml, mode="opt"):
    """
      Sets settings from input file
      @ In, xml, xml.etree.ElementTree.Element, input from user
      @ In, mode, string, case mode to operate in (e.g. 'sweep' or 'opt')
      @ Out, None
    """
    # get specs for allowable inputs
    specs = self.get_input_specs()()
    specs.parseNode(xml)
    self.name = specs.parameterValues['name']
    self.raiseADebug(f'Loading component "{self.name}"')
    for item in specs.subparts:
      if self.get_interaction() and item.getName() in ['produces', 'stores', 'demands']:
        self.raiseAnError(NotImplementedError, f'Currently each Component can only have one interaction (produces, stores, demands)! Check Component "{self.name}"')
      # read in producers
      if item.getName() == 'produces':
        prod = HeronProducer(messageHandler=self.messageHandler)
        try:
          prod.read_input(item, self.name)
        except IOError as e:
          self.raiseAWarning(f'Errors while reading component "{self.name}"!')
          raise e
        self._interaction = prod
        self._produces.append(prod)
      # read in storages
      elif item.getName() == 'stores':
        store = HeronStorage(messageHandler=self.messageHandler)
        store.read_input(item, self.name)
        self._interaction = store
        self._stores.append(store)
      # read in demands
      elif item.getName() == 'demands':
        demand = HeronDemand(messageHandler=self.messageHandler)
        demand.read_input(item, self.name)
        self._interaction = demand
        self._demands.append(demand)
      # read in economics
      elif item.getName() == 'economics':
        econ_node = item # need to read AFTER the interactions!
    # after looping over nodes, finish up
    if econ_node is None:
      self.raiseAnError(IOError, f'<economics> node missing from component "{self.name}"!')
    self._economics = HeronCashFlowGroup(self)
    self._economics.read_input(econ_node)

  def get_capacity(self, meta, raw=False):
    """
      returns the capacity of the interaction of this component
      @ In, meta, dict, arbitrary metadata from EGRET
      @ In, raw, bool, optional, if True then return the ValuedParam instance for capacity, instead of the evaluation
      @ Out, capacity, float (or ValuedParam), the capacity of this component's interaction
    """
    return self.get_interaction().get_capacity(meta, raw=raw)

  def get_uncertain_cashflow_params(self):
    """
      Get all uncertain economic parameters
      @ In, None
      @ Out, params, dict, the uncertain parameters
    """
    params = {}
    for cf in self.get_cashflows():
      uncertain = cf.get_uncertain_params()
      params |= {f"{self.name}_{k}": v for k, v in uncertain.items()}
    return params

class HeronCashFlowGroup(DoveCashFlowGroup):
  """
  """

  def read_input(self, source, xml=False):
    """
    Sets settings from input file
    @ In, source, InputData.ParameterInput, input from user
    @ In, xml, bool, if True then XML is passed in, not input data
    @ Out, None
    """
    # allow read_input argument to be either xml or input specs
    if xml:
      specs = self.get_input_specs()()
      specs.parseNode(source)
    else:
      specs = source
    # read in specs
    for item in specs.subparts:
      if item.getName() == "lifetime":
        self._lifetime = item.value
      elif item.getName() == "CashFlow":
        new = HeronCashFlow(self._component)
        new.read_input(item)
        self._cash_flows.append(new)

class HeronCashFlow(DoveCashFlow):
  """
  """
  def __repr__(self):
    """
    String representation.
    @ In, None
    @ Out, __repr__, string representation
    """
    return f'<HERON CashFlow "{self.name}">'

  # @classmethod
  # def get_input_specs(cls):
  #   """
  #     Collects input specifications for this class.
  #     @ In, None
  #     @ Out, input_specs, InputData, specs
  #   """
  #   # Grab all the DOVE input specs -- these input specs have no ValuedParams in
  #   # them, so we need to modify the input spec to allow for those VPs
  #   input_specs = super().get_input_specs()
  #   return

  def _set_value(self, name, spec):
    """
      Utilitly method to set ValuedParam members via reading input specifications.
      @ In, name, str, member variable name (e.g. self.<name>)
      @ In, spec, InputData params, input parameters
      @ Out, None
    """
    vp = ValuedParamHandler(name)
    signal = vp.read(f'CashFlow \'{self.name}\'', spec) # TODO what "mode" to use?
    self._signals.update(signal)
    self._crossrefs[name] = vp
    # standard alias: redirect "capacity" variable
    if isinstance(vp, vp_factory.returnClass('variable')) and vp.get_raven_var() == 'capacity':
      #NOTE: we are assuming here that capacity_factors are only applied in dispatch and
      # are not a variable in the outer optimization.
      vp = self._component.get_capacity_param()
    setattr(self, name, vp)

  def _set_fixed_param(self, name, value):
    """
      Fixes a ValuedParam to have a constant value
      @ In, name, str, name of member to store on "self"
      @ In, value, float, value to set for ValuedParam
      @ Out, None
    """
    vp = ValuedParamHandler(name)
    vp.set_const_VP(value)
    setattr(self, name, vp)

  def get_uncertain_params(self):
    """
      Gets any of the cashflow equation parameters which are random variables
      @ In, None
      @ Out, uncertain_params, dict[ValuedParam], the uncertain cashflow parameters
    """
    params = ["_driver", "_alpha", "_reference", "_scale"]
    uncertain_params = {}
    for param_name in params:
      if (param := getattr(self, param_name)).type == "RandomVariable":
        uncertain_params[param_name[1:]] = param
    return uncertain_params

class HeronInteraction(Base, DoveInteraction):
  """
    Base class for component interactions (e.g. Producer, Storage, Demand)
  """
  tag = 'interacts' # node name in input file

  def _set_fixed_value(self, name, value):
    """
    """
    vp = ValuedParamHandler(name)
    vp.set_const_VP(value)
    return vp

  def _set_value(self, name, comp, spec):
    """
      Sets up use of a ValuedParam for this interaction for the "name" attribute of this class.
      @ In, name, str, name of member of this class
      @ In, comp, str, name of associated component
      @ In, spec, InputParam, input specifications
      @ In, mode, string, case mode to operate in (e.g. 'sweep' or 'opt')
      @ Out, None
    """
    vp = ValuedParamHandler(name)
    signal = vp.read(comp, spec)
    self._signals.update(signal)
    self._crossrefs[name] = vp
    setattr(self, name, vp)

  def get_capacity(self, meta, raw=False):
    """
      Returns the capacity of this interaction.
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
      Returns the minimum level of this interaction.
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
  tag = 'produces' # node name in input file

  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    specs = super().get_input_specs()
    return specs
  
  def __init__(self, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    HeronInteraction.__init__(self, **kwargs)
    self._produces = []     # the resource(s) produced by this interaction
    self._consumes = []     # the resource(s) consumed by this interaction
    self._tracking_vars = ['production']

class HeronStorage(HeronInteraction, DoveStorage):
  """
    Explains a particular interaction, where a resource is stored and released later
  """
  tag = 'stores' # node name in input file

class HeronDemand(HeronInteraction, DoveDemand):
  """
    Explains a particular interaction, where a resource is demanded
  """
  tag = 'demands' # node name in input file

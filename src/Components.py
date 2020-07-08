"""
  Defines the Component entity.
"""
from __future__ import unicode_literals, print_function
import os
import sys
from collections import defaultdict
import numpy as np
from base import Base
import time
import xml.etree.ElementTree as ET
from Economics import CashFlowUser
from ValuedParams import ValuedParam
import _utils as hutils
framework_path = hutils.get_raven_loc()
sys.path.append(framework_path)
from utils import InputData, xmlUtils,InputTypes
import MessageHandler
mh = MessageHandler.MessageHandler()

def factory(xml, method='sweep'):
  """
    Tool for constructing compnents without the input_loader
    TODO can this be set up so the input_loader calls it instead of the methods directly?
    @ In, xml, ET.Element, node from which to read component settings
    @ In, method, string, optional, operational mode for case
    @ Out, comp, Component instance, component constructed
  """
  comp = Component(messageHandler=mh)
  comp.read_input(xml, method)
  return comp


class Component(Base, CashFlowUser):
  """
    Represents a unit in the grid analysis. Each component has a single "interaction" that
    describes what it can do (produce, store, demand)
  """
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    input_specs = InputData.parameterInputFactory('Component', ordered=False, baseNode=None,
        descr=r"""defines a component as an element of the grid system. Components are defined by the action they
              perform such as \xmlNode{produces} or \xmlNode{consumes}; see details below.""")
    input_specs.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""identifier for the component. This identifier will be used to generate variables
              and relate signals to this component throughout the HERON analysis.""")
    # production
    ## this unit may be able to make stuff, possibly from other stuff
    input_specs.addSub(Producer.get_input_specs())
    # storage
    ## this unit may be able to store stuff
    input_specs.addSub(Storage.get_input_specs())
    # demands
    ## this unit may have a certain demand that must be met
    input_specs.addSub(Demand.get_input_specs())
    # this unit probably has some economics
    input_specs = CashFlowUser.get_input_specs(input_specs)
    return input_specs

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, kwargs, dict, optional, arguments to pass to other constructors
      @ Out, None
    """
    Base.__init__(self, **kwargs)
    CashFlowUser.__init__(self)
    self.name = None
    self._produces = []
    self._stores = []
    self._demands = []

  def __repr__(self):
    """
      String representation.
      @ In, None
      @ Out, __repr__, string representation
    """
    return '<HERON Component "{}"">'.format(self.name)

  def read_input(self, xml, mode):
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
    self.raiseADebug('Loading component "{}"'.format(self.name))
    for item in specs.subparts:
      if self.get_interaction() and item.getName() in ['produces', 'stores', 'demands']:
        self.raiseAnError(NotImplementedError, 'Currently each Component can only have one interaction (produces, stores, demands)! Check Component "{}"'.format(self.name))
      # read in producers
      if item.getName() == 'produces':
        prod = Producer(messageHandler=self.messageHandler)
        try:
          prod.read_input(item, mode, self.name)
        except IOError as e:
          self.raiseAWarning('Errors while reading component "{}"!'.format(self.name))
          raise e
        self._produces.append(prod)
      # read in storages
      elif item.getName() == 'stores':
        store = Storage(messageHandler=self.messageHandler)
        store.read_input(item, mode, self.name)
        self._stores.append(store)
      # read in demands
      elif item.getName() == 'demands':
        demand = Demand(messageHandler=self.messageHandler)
        demand.read_input(item, mode, self.name)
        self._demands.append(demand)
      # read in economics
      elif item.getName() == 'economics':
        econ_node = item # need to read AFTER the interactions!
    # after looping over nodes, finish up
    if econ_node is None:
      self.raiseAnError(IOError, '<economics> node missing from component "{}"!'.format(self.name))
    CashFlowUser.read_input(self, econ_node)

  def get_crossrefs(self):
    """
      Collect the required value entities needed for this component to function.
      @ In, None
      @ Out, crossrefs, dict, mapping of dictionaries with information about the entities required.
    """
    inter = self.get_interaction()
    crossrefs = {inter: inter.get_crossrefs()}
    crossrefs.update(self._economics.get_crossrefs())
    return crossrefs

  def set_crossrefs(self, refs):
    """
      Connect cross-reference material from other entities to the ValuedParams in this component.
      @ In, refs, dict, dictionary of entity information
      @ Out, None
    """
    try_match = self.get_interaction()
    for interaction in list(refs.keys()):
      # find associated interaction
      if try_match == interaction:
        try_match.set_crossrefs(refs.pop(interaction))
        break
    # send what's left to the economics
    self._economics.set_crossrefs(refs)
    # if anything left, there's an issue
    assert not refs

  def get_interaction(self):
    """
      Return the interactions this component uses.
      TODO could this just return the only non-empty one, since there can only be one?
      @ In, None
      @ Out, interactions, list, list of Interaction instances
    """
    try:
      return (self._produces + self._stores + self._demands)[0]
    except IndexError: # there are no interactions!
      return None

  def print_me(self, tabs=0, tab='  '):
    """
      Prints info about self
      @ In, tabs, int, optional, number of tabs to insert before prints
      @ In, tab, str, optional, characters to use to denote hierarchy
      @ Out, None
    """
    pre = tab*tabs
    print(pre+'Component:')
    print(pre+'  name:', self.name)
    self.get_interaction().print_me(tabs=tabs+1, tab=tab)

  def get_inputs(self):
    """
      returns list of all resources consumed here
      @ In, None
      @ Out, inputs, set, set of input resources as strings (resources that are taken/consumed/stored)
    """
    inputs = set()
    # simply combine the inputs for the interaction
    inputs.update(self.get_interaction().get_inputs())
    return inputs

  def get_outputs(self):
    """
      returns list of all resources producable here
      @ In, None
      @ Out, outputs, set, set of output resources as strings (resources that are produced/provided)
    """
    outputs = set()
    outputs.update(self.get_interaction().get_outputs())
    return outputs

  def get_resources(self):
    """
      Provides the full set of resources used by this component.
      @ In, None
      @ Out, res, set, set(str) of resource names
    """
    res = set()
    res.update(self.get_inputs())
    res.update(self.get_outputs())
    return res

  def get_capacity(self, meta, raven_vars, dispatch, t, raw=False):
    """
      returns the capacity of the interaction of this component
      @ In, meta, dict, arbitrary metadata from EGRET
      @ In, raven_vars, dict, evaluated RAVEN variables
      @ In, dispatch, DispatchScenario.DispatchRecord, current dispatch situation
      @ In, t, int, current time step
      @ In, raw, bool, optional, if True then return the ValuedParam instance for capacity, instead of the evaluation
      @ Out, capacity, float (or ValuedParam), the capacity of this component's interaction
    """
    return self.get_interaction().get_capacity(meta, raven_vars, dispatch, t, raw=raw)

  def get_capacity_var(self):
    """
      Returns the variable that is used to define this component's capacity.
      @ In, None
      @ Out, var, str, name of capacity resource
    """
    return self.get_interaction().get_capacity_var()

  def is_dispatchable(self):
    """
      Returns the dispatchability indicator of this component.
      TODO Note that despite the name, this is NOT boolean, but a string indicator.
      @ In, None
      @ Out, dispatchable, str, dispatchability (e.g. independent, dependent, fixed)
    """
    return self.get_interaction().is_dispatchable()

  def set_capacity(self, cap):
    """
      Set the float value of the capacity of this component's interaction
      @ In, cap, float, value
      @ Out, None
    """
    return self.get_interaction().set_capacity(cap)

  def produce(self, request, meta, raven_variables, dispatch, t, level=None):
    """
      Enacts the transfer function for this component to act based on a request.
      FIXME was used for "generic" dispatcher, does it still apply?
      @ In, request, dict, mapping of requested resource usage to amount requested (negative is
                           consume, positive is produce)
      @ In, meta, dict, metadata information for current status in run
      @ In, raven_variables, dict, variables from RAVEN TODO part of meta!
      @ In, dispatch, DispatchState, expression of the current activity levels in the system
      @ In, t, int, index of "time" at which this production should be performed
      @ In, level, float, for storages indicates the amount currently stored
      @ Out, balance, dict, full dict of resources used and produced for request
      @ Out, meta, dict, updated metadata dictionary
    """
    #balance = defaultdict(float)
    interaction = self.get_interaction()
    balance, meta = interaction.produce(request, meta, raven_variables, dispatch, t, level)
    #for resource, quantity in int_balance.items():
    #  balance[resource] += quantity
    return balance, meta

  def produce_max(self, meta, raven_variables, dispatch, t):
    """
      Determines the maximum production possible for this component.
      @ In, meta, dict, metadata information for current status in run
      @ In, raven_variables, dict, variables from RAVEN TODO part of meta!
      @ In, dispatch, DispatchState, expression of the current activity levels in the system
      @ In, t, int, index of "time" at which this production should be performed
      @ Out, balance, dict, full dict of resources used and produced for request
      @ Out, meta, dict, updated metadata dictionary
    """
    #balance = defaultdict(float)
    interaction = self.get_interaction()
    balance, meta = interaction.produce_max(meta, raven_variables, dispatch, t)
    #for resource, quantity in int_balance.items():
    #  balance[resource] += quantity
    return balance, meta

  def produce_min(self, meta, raven_variables, dispatch, t):
    """
      Determines the minimum production possible for this component.
      @ In, meta, dict, metadata information for current status in run
      @ In, raven_variables, dict, variables from RAVEN TODO part of meta!
      @ In, dispatch, DispatchState, expression of the current activity levels in the system
      @ In, t, int, index of "time" at which this production should be performed
      @ Out, balance, dict, full dict of resources used and produced for request
      @ Out, meta, dict, updated metadata dictionary
    """
    #balance = defaultdict(float)
    interaction = self.get_interaction()
    balance, meta = interaction.produce_min(meta, raven_variables, dispatch, t)
    #for resource, quantity in int_balance.items():
    #  balance[resource] += quantity
    return balance, meta

  def get_capacity_param(self):
    """
      Provides direct access to the ValuedParam for the capacity of this component.
      @ In, None
      @ Out, cap, ValuedParam, capacity valued param
    """
    intr = self.get_interaction()
    return intr.get_capacity(None, None, None, None, raw=True)







class Interaction(Base):
  """
    Base class for component interactions (e.g. Producer, Storage, Demand)
  """
  tag = 'interacts' # node name in input file

  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    if cls.tag == 'produces':
      desc = r"""indicates that this component produces one or more resources by consuming other resources."""
      resource_desc = r"""the resource produced by this component's activity."""
    elif cls.tag == 'stores':
      desc = r"""indicates that this component stores one resource, potentially absorbing or providing that resource."""
      resource_desc = r"""the resource stored by this component."""
    elif cls.tag == "demands":
      desc = r"""indicates that this component exclusively consumes a resource."""
      resource_desc = r"""the resource consumed by this component."""
    specs = InputData.parameterInputFactory(cls.tag, ordered=False, descr=desc)
    specs.addParam('resource', param_type=InputTypes.StringListType, required=True,
        descr=resource_desc)
    dispatch_opts = InputTypes.makeEnumType('dispatch_opts', 'dispatch_opts', ['fixed', 'independent', 'dependent'])
    specs.addParam('dispatch', param_type=dispatch_opts,
        descr=r"""describes the way this component should be dispatched, or its flexibility.
              \texttt{fixed} indicates the component always fully dispatched at its maximum level.
              \texttt{independent} indicates the component is fully dispatchable by the dispatch optimization algorithm.
              \texttt{dependent} indicates that while this component is not directly controllable by the dispatch
              algorithm, it can however be flexibly dispatched in response to other units changing dispatch level.
              For example, when attempting to increase profitability, the \texttt{fixed} components are not adjustable,
              but the \texttt{independent} components can be adjusted to attempt to improve the economic metric.
              In response to the \texttt{independent} component adjustment, the \texttt{dependent} components
              may respond to balance the resource usage from the changing behavior of other components.""")

    cap = ValuedParam.get_input_specs('capacity')
    cap.descr = r"""provides the maximum value at which this component can act, in units of the indicated resource. """
    #cap.removeSub('ARMA')
    #cap.removeSub('Function')
    #cap.removeSub('variable')
    cap.addParam('resource', param_type=InputTypes.StringType,
        descr=r"""indicates the resource that defines the capacity of this component's operation. For example,
              if a component consumes steam and electricity to produce hydrogen, the capacity of the component
              can be defined by the maximum steam consumable, maximum electricity consumable, or maximum
              hydrogen producable. Any choice should be nominally equivalent, but determines the units
              of the value of this node.""")
    specs.addSub(cap)

    minn = ValuedParam.get_input_specs('minimum')
    minn.descr = r"""provides the minimum value at which this component can act, in units of the indicated resource. """
    minn.addParam('resource', param_type=InputTypes.StringType,
        descr=r"""indicates the resource that defines the minimum activity level for this component,
              as with the component's capacity.""")
    specs.addSub(minn)
    return specs

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, kwargs, dict, arbitrary pass-through arguments
      @ Out, None
    """
    Base.__init__(self, **kwargs)
    self._capacity = None               # upper limit of this interaction
    self._capacity_var = None           # which variable limits the capacity (could be produced or consumed?)
    self._signals = set()               # dependent signals for this interaction
    self._crossrefs = defaultdict(dict) # crossrefs objects needed (e.g. armas, etc), as {attr: {tag, name, obj})
    self._dispatchable = None           # independent, dependent, or fixed?
    self._minimum = None                # lowest interaction level, if dispatchable
    self._minimum_var = None            # limiting variable for minimum
    self._function_method_map = {}      # maps things that call functions to the method within the function that needs calling
    self._transfer = None               # the production rate (if any), in produces per consumes
                                        #   for example, {(Producer, 'capacity'): 'method'}

  def read_input(self, specs, mode, comp_name):
    """
      Sets settings from input file
      @ In, specs, InputData, specs
      @ Out, None
    """
    self.raiseADebug(' ... loading interaction "{}"'.format(self.tag))
    self._dispatchable = specs.parameterValues['dispatch']
    for item in specs.subparts:
      name = '_' + item.getName()
      if name in ['_capacity', '_minimum']:
        # common reading for valued params
        self._set_valued_param(name, comp_name, item, mode)
        if name == '_capacity':
          self._capacity_var = item.parameterValues.get('resource', None)
        elif item.getName() == 'minimum':
          self._minimum_var = item.parameterValues.get('resource', None)
    # finalize some values
    resources = set(list(self.get_inputs()) + list(self.get_outputs()))
    ## capacity: if "variable" is None and only one resource in interactions, then that must be it
    if self._capacity_var is None:
      if len(resources) == 1:
        self._capacity_var = list(resources)[0]
      else:
        self.raiseAnError(IOError, 'If multiple resources are active, "capacity" requires a "resource" specified!')
    ## minimum: basically the same as capacity, functionally
    if self._minimum and self._minimum_var is None:
      if len(resources) == 1:
        self._minimum_var = list(resources)[0]
      else:
        self.raiseAnError(IOError, 'If multiple resources are active, "minimum" requires a "resource" specified!')

  def _set_valued_param(self, name, comp, spec, mode):
    """
      Sets up use of a ValuedParam for this interaction for the "name" attribute of this class.
      @ In, name, str, name of member of this class
      @ In, comp, str, name of associated component
      @ In, spec, InputParam, input specifications
    """
    vp = ValuedParam(name)
    signal = vp.read(comp, spec, mode)
    self._signals.update(signal)
    self._crossrefs[name] = vp
    setattr(self, name, vp)

  def get_capacity(self, meta, raven_vars, dispatch, t, raw=False):
    if raw:
      return self._capacity
    request = {self._capacity_var: None}
    inputs = {'request': request,
              'meta': meta,
              'raven_vars': raven_vars,
              'dispatch': dispatch,
              't': t}
    evaluated, meta = self._capacity.evaluate(inputs, target_var=self._capacity_var)
    return evaluated, meta

  def get_capacity_var(self):
    return self._capacity_var

  def set_capacity(self, cap):
    self._capacity.type = 'value'
    self._capacity._value = float(cap) # TODO getter/setter

  def get_minimum(self, meta, raven_vars, dispatch, t, raw=False):
    if raw:
      return self._minimum
    request = {self._minimum_var: None}
    inputs = {'request': request,
              'meta': meta,
              'raven_vars': raven_vars,
              'dispatch': dispatch,
              't': t}
    evaluated, meta = self._minimum.evaluate(inputs, target_var=self._minimum_var)
    return evaluated, meta

  def get_crossrefs(self):
    return self._crossrefs

  def set_crossrefs(self, refs):
    #assert set(refs.keys()) == set(self._crossrefs.keys()), 'Interaction "{}": Requests and provided cross-references do not match! refs: {} reqs: {}'.format(self.tag, list(refs.keys()), list(self._crossrefs.keys()))
    for attr, obj in refs.items():
      valued_param = self._crossrefs[attr]
      valued_param.set_object(obj)

  def get_inputs(self, dependencies=False):
    if dependencies:
      xxxxxxxxx # TODO do I use this? Answer is no, not currently.
      return set(self._signals)
    else:
      return set()

  def get_outputs(self):
    return set()

  def get_resources(self):
    return list(self.get_inputs()) + list(self.get_outputs())

  def is_dispatchable(self):
    return self._dispatchable

  def is_type(self, typ):
    return typ == self.__class__.__name__

  def produce(self, *args, **kwargs):
    raise NotImplementedError('This interaction has no "produce" method.')

  def produce_max(self, *args, **kwargs):
    raise NotImplementedError('This interaction has no produce_max method yet!')

  def produce_min(self, *args, **kwargs):
    raise NotImplementedError('This interaction has no produce_min method yet!')

  def check_expected_present(self, data, expected, premessage):
    """ checks dict to make sure members are present and not None """
    # check missing
    missing = list(d for d in expected if d not in data)
    if missing:
      self.raiseAWarning(premessage, '| Expected variables are missing:', missing)
    # check None
    nones = list(d for d, v in data.items() if (v is None and v in expected))
    if nones:
      self.raiseAWarning(premessage, '| Expected variables are None:', nones)
    if missing or nones:
      self.raiseAnError(RuntimeError, 'Some variables were missing or None! See warning messages above for details!')

  def _check_capacity_limit(self, res, amt, balance, meta, raven_vars, dispatch, t):
    cap = self.get_capacity(meta, raven_vars, dispatch, t)[0][self._capacity_var]
    try:
      if abs(balance[self._capacity_var]) > abs(cap):
        #ttttt
        # do the inverse problem: how much can we make?
        balance, meta = self.produce_max(meta, raven_vars, dispatch, t)
        print('The full requested amount ({res}: {req}) was not possible, so accessing maximum available instead ({res}: {blc}).'.format(res=res, req=amt, blc=balance[res]))
    except KeyError:
      raise SyntaxError('Resource "{}" is listed as capacity limiter, but not an output of the component! Got: {}'.format(self._capacity_var, balance))
    return balance, meta

  def get_transfer(self):
    """ Returns the transfer function, if any TODO """
    return self._transfer



class Producer(Interaction):
  """
    Explains a particular interaction, where a resource is consumed to produce another resource
  """
  tag = 'produces' # node name in input file

  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    specs = super(Producer, cls).get_input_specs()
    specs.addSub(InputData.parameterInputFactory('consumes', contentType=InputTypes.StringListType, descr=r"""The producer can either produce or consume a resource. If the producer is a consumer it must be accompnied with a transfer function to convert one source of energy to another. """))
    specs.addSub(ValuedParam.get_input_specs('transfer'))
    return specs

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Interaction.__init__(self, **kwargs)
    self._produces = []   # the resource(s) produced by this interaction
    self._consumes = []   # the resource(s) consumed by this interaction

  def read_input(self, specs, mode, comp_name):
    """
      Sets settings from input file
      @ In, specs, InputData, specs
      @ Out, None
    """
    # specs were already checked in Component
    Interaction.read_input(self, specs, mode, comp_name)
    self._produces = specs.parameterValues['resource']
    for item in specs.subparts:
      if item.getName() == 'consumes':
        self._consumes = item.value
      elif item.getName() == 'transfer':
        self._set_valued_param('_transfer', comp_name, item, mode)

    # input checking
    ## if a transfer function not given, can't be consuming a resource
    if self._transfer is None:
      if self._consumes:
        self.raiseAnError(IOError, 'Any component that consumes a resource must have a transfer function describing the production process!')
    #else if transfer function is a float/ARMA, then there must be only one output, one input
    else:
      if self._transfer.type in ['value', 'ARMA']: #isinstance(self._transfer, float) or self._transfer['type'] == 'ARMA':
        if not (len(self.get_inputs()) == 1 and len(self.get_outputs() == 1)):
          self.raiseAnError(IOError, 'Transfer function ("<transfer>") can be a float/ARMA only if the component '+\
                                     'produces exactly one resource and consumes exactly one resource!\n' +\
                                     '    Consumes: {}'.format(self.get_inputs()) +\
                                     '    Produces: {}'.format(self.get_outputs()))

  def get_inputs(self):
    inputs = Interaction.get_inputs(self)
    inputs.update(np.atleast_1d(self._consumes))
    return inputs

  def get_outputs(self):
    outputs = set(np.atleast_1d(self._produces))
    return outputs

  def print_me(self, tabs=0, tab='  '):
    """ Prints info about self """
    pre = tab*tabs
    print(pre+'Producer:')
    print(pre+'  produces:', self._produces)
    print(pre+'  consumes:', self._consumes)
    print(pre+'  transfer:', self._transfer)
    print(pre+'  capacity:', self._capacity)

  def produce(self, request, meta, raven_vars, dispatch, t, level=None):
    # is there a transfer function to apply?
    res, amt = next(iter(request.items()))
    if self._transfer:
      balance, meta = self.transfer(request, meta, raven_vars, dispatch, t)
    # TODO catch errors! Custom user-based errors?
    else:
      # no transfer function, then we provide the request as-is
      balance = request
    # check if capacity was exceeded
    balance, meta = self._check_capacity_limit(res, amt, balance, meta, raven_vars, dispatch, t)
    return balance, meta

  def produce_max(self, meta, raven_vars, dispatch, t):
    request, meta = self.get_capacity(meta, raven_vars, dispatch, t)
    balance, meta = self.produce(request, meta, raven_vars, dispatch, t)
    # dict((prod, self._capacity[p]) for p, prod in enumerate(self._produces))
    return balance, meta

  def produce_min(self, meta, raven_vars, dispatch, t):
    if self._minimum:
      request, meta = self.get_minimum(meta, raven_vars, dispatch, t)#[self._minimum]
      request = {self._minimum_var: request[self._minimum_var]}
    else:
      request = {next(iter(self.get_outputs())): 0.0}
    balance, meta = self.produce(request, meta, raven_vars, dispatch, t)
    return balance, meta

  def transfer(self, request, meta, raven_vars, dispatch, t):
    """ Use the transfer function to make a balance of activities that should occur """
    assert len(request) == 1
    balance = defaultdict(float)
    # in the rare case that the transfer function is simple ...
    resources_in = list(self.get_inputs())
    resources_out = list(self.get_outputs())
    inputs = {'request': request,
              'meta': meta,
              'raven_vars': raven_vars,
              'dispatch': dispatch,
              't': t}
    balance, meta = self._transfer.evaluate(inputs)
    self.check_expected_present(balance, self.get_resources(), 'TRANSFER FUNCTION {}'.format(self._transfer))
    # OLD if transfer evaluation is a float (float, arma), then it signifies a conversion rate
    ## note that we've checked in the input reading for this singular relationship
    print(balance)

    if False: #len(balance) == 1:
      requested, rate = list(balance.items())[0] # requested resource and the transfer rate (amount of product per consumed)
      amount = list(requested.values())[0]       # amount of requested resource
      if requested in resources_in:
        balance[resources_out[0]] = -1.0 * rate * amount # NOTE: amount should be negative, but output should be positive
      else:
        balance[inputs[0]] = -1.0 / rate * amount  # NOTE: amount should be positive, but input should be negative
    # check that all values got filled -> TODO remove this for opt performance
    missing = set(resources_in + resources_out) - set(balance.keys())
    if missing:
      self.raiseAnError(RuntimeError, 'While evaluating transfer function, not all variables requested were provided!' +\
                        '  Missing: {}'.format(missing) +\
                        '  Transfer function: {}'.format(self._transfer))
    return balance, meta





class Storage(Interaction):
  """
    Explains a particular interaction, where a resource is stored and released later
  """
  tag = 'stores' # node name in input file

  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    specs = super(Storage, cls).get_input_specs()
    specs.addSub(ValuedParam.get_input_specs('rate'))
    specs.addSub(ValuedParam.get_input_specs('initial_stored'))
    return specs

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Interaction.__init__(self, **kwargs)
    self._stores = None   # the resource stored by this interaction
    self._rate = None     # the rate at which this component can store up or discharge
    self._initial_stored = None # how much resource does this component start with stored?

  def read_input(self, specs, mode, comp_name):
    """
      Sets settings from input file
      @ In, specs, InputData, specs
      @ Out, None
    """
    # specs were already checked in Component
    Interaction.read_input(self, specs, mode, comp_name)
    self._stores = specs.parameterValues['resource']
    for item in specs.subparts:
      if item.getName() == 'rate':
        self._set_valued_param('_rate', comp_name, item, mode)
      elif item.getName() == 'initial_stored':
        self._set_valued_param('_initial_stored', comp_name, item, mode)
    assert len(self._stores) == 1, 'Multiple storage resources given for component "{}"'.format(comp_name)
    self._stores = self._stores[0]
    # checks and defaults
    if self._initial_stored is None:
      self.raiseAWarning('Initial storage level for "{}" was not provided! Defaulting to 0.'.format(comp_name))
      # make a fake reader node for a 0 value
      vp = ValuedParam('initial_stored')
      vp.type = 'value'
      vp._value = 0.0 # TODO getter/setter, also a better default value setting?
      self._initial_stored = vp
    # the capacity is limited by the stored resource.
    self._capacity_var = self._stores

  def get_inputs(self):
    inputs = Interaction.get_inputs(self)
    inputs.update(np.atleast_1d(self._stores))
    return inputs

  def get_outputs(self):
    outputs = Interaction.get_outputs(self)
    outputs.update(np.atleast_1d(self._stores))
    return outputs

  def get_resource(self):
    return self._stores

  def print_me(self, tabs=0, tab='  '):
    """ Prints info about self """
    pre = tab*tabs
    print(pre+'Storage:')
    print(pre+'  stores:', self._stores)
    print(pre+'  rate:', self._rate)
    print(pre+'  capacity:', self._capacity)

  def produce(self, request, meta, raven_vars, dispatch, t, level=None):
    if level is None:
      raise RuntimeError('Storage level information was not provided to Storage produce call!')
    dt = dispatch()['time'].values
    dt = dt[t] - dt[t-1] if t > 0 else dt[t+1] - dt[t]
    res, amt_rate = next(iter(request.items()))
    # UNITS: amt_rate is in resource per time, NOT pure resource!
    amt_amount = amt_rate * dt
    assert res == self.get_resource(), 'Requested var is not the same as stored var!'
    balance, meta = self._check_capacity_limit(res, amt_amount, {}, meta, raven_vars, dispatch, t, level)
    # also check rate limit
    delta_amount = balance[res]
    delta_rate = delta_amount / dt # flip the sign to show we're doing as we were asked ...?
    #print('DEBUGG effective battery cons/prod rate:', delta_rate)
    balance, meta = self._check_rate_limit(res, delta_rate, {}, meta, raven_vars, dispatch, t)
    return balance, meta

  def _check_capacity_limit(self, res, amt, balance, meta, raven_vars, dispatch, t, level):
    """ overloads Interaction method, since units for storage are "res" not "res per second" """
    # note "amt" has units of AMOUNT not RATE (resource, not resource per second)
    sign = np.sign(amt)
    # are we storing or providing?
    #print('DEBUGG supposed current level:', level)
    if sign < 0:
      # we are being asked to consume some
      cap, meta = self.get_capacity(meta, raven_vars, dispatch, t)
      available_amount = cap[res] - level
      #print('Supposed Capacity, Only calculated ins sign<0 (being asked to consumer)',cap)
    else:
      # we are being asked to produce some
      available_amount = level
    # the amount we can consume is the minimum of the requested or what's available
    delta = sign * min(available_amount, abs(amt))
    print('available_amount', available_amount)
    print('request asked for', amt)
    print('end provision:', delta)
    return {res: delta}, meta

  def _check_rate_limit(self, res, amt, balance, meta, raven_vars, dispatch, t):
    # TODO distinct up/down rates
    # check limiting rate for resource flow in/out, if any
    if self._rate:
      request = {res: None}
      inputs = {'request': request,
                'meta': meta,
                'raven_vars': raven_vars,
                'dispatch': dispatch,
                't': t}
      max_rate = self._rate.evaluate(inputs, target_var=res)[0][res]
      delta = np.sign(amt) * min(max_rate, abs(amt))
      print('max_rate in _check_rate_limit',max_rate, 'delta (min of maxrate and abs(amt)',delta)
      return {res: delta}, meta
    return {res: amt}, meta

  def produce_max(self, meta, raven_vars, dispatch, t):
    # TODO
    return {}, meta

  def produce_min(self, meta, raven_vars, dispatch, t):
    # TODO
    return {}, meta

  def get_initial_level(self, meta, raven_vars, dispatch, t):
    res = self.get_resource()
    request = {res: None}
    inputs = {'request': request,
              'meta': meta,
              'raven_vars': raven_vars,
              'dispatch': dispatch,
              't': t}
    return self._initial_stored.evaluate(inputs, target_var=res)[0][res]



class Demand(Interaction):
  """
    Explains a particular interaction, where a resource is demanded
  """
  tag = 'demands' # node name in input file

  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    specs = super(Demand, cls).get_input_specs()
    specs.addSub(ValuedParam.get_input_specs('penalty'))
    return specs

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Interaction.__init__(self, **kwargs)
    self._demands = None  # the resource demanded by this interaction
    self._penalty = None  # how to penalize for not meeting demand NOT IMPLEMENTED

  def read_input(self, specs, mode, comp_name):
    """
      Sets settings from input file
      @ In, specs, InputData, specs
      @ Out, None
    """
    # specs were already checked in Component
    # must set demands first, so that "capacity" can access it
    self._demands = specs.parameterValues['resource']
    Interaction.read_input(self, specs, mode, comp_name)
    for item in specs.subparts:
      if item.getName() == 'penalty':
        self._set_valued_param('_rate', comp_name, item, mode)

  def get_inputs(self):
    inputs = Interaction.get_inputs(self)
    inputs.update(np.atleast_1d(self._demands))
    return inputs

  def print_me(self, tabs=0, tab='  '):
    """ Prints info about self """
    pre = tab*tabs
    print(pre+'Demand/Load:')
    print(pre+'  demands:', self._demands)
    print(pre+'  penalty:', self._penalty)
    print(pre+'  capacity:', self._capacity)

  def produce(self, request, meta, raven_vars, dispatch, t, level=None):
    # Q: should this have a transfer function or something? At least capacity limits?
    # A: no; if you want this functionality, add an intervening component with a transfer function.
    res, amt = next(iter(request.items()))
    balance, meta = self._check_capacity_limit(res, amt, request, meta, raven_vars, dispatch, t)
    res, amt = next(iter(balance.items()))
    amt = -1 * abs(amt)
    balance[res] = amt
    return balance, meta

  def produce_max(self, meta, raven_vars, dispatch, t):
    request, meta = self.get_capacity(meta, raven_vars, dispatch, t)
    return request, meta

  def produce_min(self, meta, raven_vars, dispatch, t):
    if self._minimum:
      request, meta = self.get_minimum(meta, raven_vars, dispatch, t)
    else:
      request = {next(iter(self.get_inputs())): 0.0} # TODO is this a good choice when no min var avail?
    return request, meta
#if __name__=="__main__":
  #comp = Component()
  #print(comp.__repr__)
#  help(Producer)
#  comp = Component()
#  Component.__repr__
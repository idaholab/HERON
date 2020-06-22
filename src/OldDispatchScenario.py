"""
  Helper classes for tracking branching dispatch scenarios
"""
import copy
import numpy as np
import pandas as pd
import xarray as xr
import pprint
import time as time_mod

class Scenario:
  def __init__(self, name, state, parent=None, pinned=None):
    #name, dispatch, storages, resource_map, meta, raven_vars, dispatchable, dependent, time, t, pinned=None):
    # Input Values
    self.name = name
    # to which scenario does this scenario belong?
    self.parent = parent
    ## dispatch of resources by units when starting this scenario, including past time steps
    self.original_dispatch = state['dispatch']
    ## list of dispatchable components who can be perturbed
    self.dispatchable = state['components']['independent']
    ## list of dependent components that can respond to perturbations but are not directly perturbable
    self.dependent = state['components']['dependent']
    ## list of fixed components that have no flexibility
    self.fixed = state['components']['fixed']
    ## storage units
    self.storages = state['components']['storages']
    ## map of resources to the components that consume and produce them
    self.resource_map = state['resource map']
    ## state of meta variables when starting this scenario
    self._meta = copy_meta(state['meta'])
    ## curent value of RAVEN variables
    self.raven_vars = state['raven vars']
    ## times corresponding to dispatch columns
    self._time = state['time window']
    ## integer time step of interest to this scenario
    self._t = state['initial step number']
    ## components pinned at the start of this scenario; pinned components cannot be additionally dispatched
    self.pinned = pinned if pinned is not None else []

    # scenario branches under this scenario that are worth tracking.
    self._sub_scenarios = {}

    # Derived Values
    self.resources = sorted(list(self.resource_map.keys()))
    if isinstance(self.dispatchable[0], str):
      #print(self.dispatchable)
      #time_mod.sleep(2000000)
      self.disp_names = self.dispatchable + self.dependent + self.fixed
    else:
      self.disp_names = list(x.name for x in self.dispatchable + self.dependent + self.fixed)
    ## new dispatch as a result of this scenario; full dispatch is original_dispatch + add_dispatch
    self.add_dispatch = DispatchRecord(self.resources, self.disp_names, self._time)
    ## new storage levels in this scenario -> should this be dynamically calculated?
    # TODO FIXME XXX
    ## perturbations that should be removed from the total balance but not local balance
    self._total_ptbs = DispatchRecord(self.resources, ['perturb_req'], self._time)# TODO FIXME WORKING
    ## calculated cost of run
    self.cost = None

  def get_time(self, t):
    return self._time[t - self._t]

  def get_start_time(self):
    return self.get_time(self.get_start_step())

  def get_end_time(self):
    return self.get_time(self.get_end_step())

  def get_time_steps(self):
    return self._time

  def get_start_step(self):
    return self._t

  def get_end_step(self):
    return self._t + len(self._time) - 1

  def get_balance(self, request, ignore_perturbs=False):
    """return balance depending on the request"""
    # if the data is provided, use that to make the balance
    if isinstance(request, DispatchRecord):
      return request.make_balance()
    # otherwise, need to make the dispatch first before making the balance
    if request == 'total':
      # total balance including original + new from this scenario so far
      data = self.original_dispatch + self.add_dispatch
    elif request == 'delta':
      # just the new balance as a result of this scnario
      data = self.add_dispatch
      if not ignore_perturbs:
        data += self._total_ptbs
    elif request == 'original':
      # the original balance before this scenario
      data = self.original_dispatch
    else:
      raise RuntimeError('Unrecognized request:', request)
    return data.make_balance()

  def get_parents_deltas(self):
    """ recursively get the deltas for each parent going up the tree """
    if self.parent is not None:
      delta = self.parent.get_balance('delta')
      # XXX FIXME what's the COST of this parent's delta?????
      # TODO I think only the "dispatch" knows how to answer that.
      more = self.parent.get_parents_deltas()
      if more is not None:
        delta = delta + more
      return delta
    return None

  def get_dispatch(self, request):
    """return the dispatch depending on the request"""
    if request == 'total':
      # TODO speedup, check if there's anything in add_dispatch before combining!
      # total dispatch including original plus new
      ret = self.original_dispatch + self.add_dispatch
    elif request == 'delta':
      # dispatch only as a result of this scenario
      ret = self.add_dispatch
    elif request == 'original':
      # dispatch before considering this scenario
      ret = self.original_dispatch
    else:
      raise RuntimeError('Unrecognized request:', request)
    return ret

  def get_storage_levels(self, request, request_comp=None):
    """ use the dispatch and initial condition to determine the storage levels of the storages """
    if request_comp is None:
      comps = self.storages
      ret_dict = True
    else:
      comps = [request_comp]
      ret_dict = False
    names = list(x.name for x in comps)
    # TODO t should be passed in, not assumed!
    assert len(self.get_time_steps()) == 1
    dispatch = self.get_dispatch(request)
    initials = {}
    resources = {}
    for comp in comps:
      name = comp.name
      storage = comp.get_interaction()
      initials[name] = storage.get_initial_level(self.get_meta(dispatch=request), self.raven_vars, dispatch, self._t)
      resources[name] = storage.get_resource()
    quantities = dispatch.get_net_quantity(names, initials, resources)
    # if no particular comp was requested, return all the levels
    if ret_dict:
      return quantities
    # otherwise, return just the float value of the level of the requested comp
    return quantities[comp.name]

  def get_comp_dispatch(self, comp, source):
    #loc = {'time': self.time[self.t], 'component': comp}
    # TODO time should be sent in, not assumed!!
    assert len(self.get_time_steps()) == 1
    time = self.get_time(self._t)
    if source in ['original', 'total']:
      original = self.get_dispatch('original').get_comp_dispatch(time, comp) # loc[loc].drop(['time', 'component'])
    if source in ['delta', 'total']:
      delta = self.get_dispatch('delta').get_comp_dispatch(time, comp) #.loc[loc].drop(['time', 'component'])
    if source == 'original':
      tot = original
    elif source == 'total':
      tot = original + delta
    elif source == 'delta':
      tot = delta
    d = dict((res, float(tot.loc[{'resource':res}])) for res in tot.coords['resource'].values)
    return d

  def get_unbalanced(self, source):
    balance = self.get_balance(source)
    return sorted(balance[abs(balance)>1e-10].dropna(axis='index').index.values) # TODO hardcorded threshold!

  def add_branch(self, ident, pin):
    """add a sub-scenario to this scenario"""
    # don't copy all the meta; the EGRET meta should be reference not deepcopied
    ident = '{}.{}'.format(self.name, ident)
    meta = copy_meta(self._meta)
    pinned = self.pinned + pin
    # getting the dispatch then making the balance saves 1 call to get_dispatch('total') which is quite slow.
    # -> the savings is because we need both the dispatch and the balance
    new_base_dispatch = self.get_dispatch('total')
    #new_base_balance = self.get_balance(new_base_dispatch)
    state = {'dispatch': new_base_dispatch,
             'components': {'independent': self.dispatchable,
                            'dependent': self.dependent,
                            'fixed': self.fixed,
                            'storages': self.storages},
             'resource map': self.resource_map,
             'meta': meta,
             'raven vars': self.raven_vars,
             'time window': self._time,
             'initial step number': self._t,
    }
    new = Scenario(ident, state, parent=self, pinned=pinned)
    self._sub_scenarios[ident] = new
    return new

  def remove_branch(self, branch):
    return self._sub_scenarios.pop(branch.name)

  def clear_branches(self):
    self._sub_scenarios = {}

  def get_branches(self):
    return self._sub_scenarios

  def get_meta(self, dispatch=None, make_copy=True):
    if make_copy:
      meta = copy_meta(self._meta)
    else:
      meta = self._meta
    meta['EGRET']['time'] = self._time
    if dispatch is not None:
      meta['EGRET']['dispatch'] = self.get_dispatch(dispatch)
    return meta

  def collapse_branches(self, name):
    """ one branch has been chosen; roll the changes up onto the main line """
    branch = self._sub_scenarios[name]
    self.merge(branch)
    self._sub_scenarios = {}

  def merge(self, other):
    """ merge another scenario onto this one """
    # update meta
    self.update_meta(other.get_meta(dispatch=None))
    # combine differential dispatch
    self.add_dispatch += other.add_dispatch

  def update_meta(self, new_meta):
    self._meta.update(new_meta)

  def dispatch(self, changes, comp, t):
    if isinstance(comp, str):
      assert comp == 'parent'
      # this means we will be borrowing from the parent, so store it in perturbs
      for res, qty in changes.items():
        if qty != 0.0:
          self.perturb_balance(res, qty)
    else:
      for res, qty in changes.items():
        if qty != 0.0:
          #print('')
          #print('DEBUGG dispatching change, SCENARIO:', self.name)
          if comp.get_interaction().is_type('Storage'):
            #print('I\'m a STORAGE I think', comp.name)
            #print('AMOUNT to dispatch:', res, qty)
            #print('INITIAL storage:', self.get_storage_levels('original', request_comp=comp))
            #print('CURRENT storage:', self.get_storage_levels('total', request_comp=comp))
            self.get_dispatch('delta').add(res, comp.name, self.get_time(t), qty)
            #print('AFTER storage:', self.get_storage_levels('total', request_comp=comp))
            #print('')
          else:
            #print('I\'m a STORAGE I think NOT', comp.name)
            self.get_dispatch('delta').set(res, comp, self.get_time(t), qty)

  def perturb_balance(self, resource, amount):
    self._total_ptbs.add(resource, 'perturb_req', self.get_time(self._t), amount)
    #self._total_ptbs.add('perturb_req', resource, amount, self.get_time(self._t))

  def get_perturbed(self, resource):
    try:
      qty = float(self._total_ptbs.get_comp_dispatch(self.get_time(self._t), 'perturb_req').loc[{'resource': resource}])
    except KeyError as e:
      self._total_ptbs.prettyprint()
      raise e
    return qty

  def set_cost(self, cost):
    self.cost = cost

  def accept_delta(self):
    """ merge delta into original """
    self.original_dispatch += self.add_dispatch
    self.add_dispatch = DispatchRecord(self.resources, self.disp_names, self._time)





class DispatchRecord:
  # NOTE Pandas dataframes were almost twice as fast for data containing only component and resource
  #   when compared to xarray dataarrays, IF "time" wasn't included.
  #   However, when adding "time", XArray became 33% faster.
  def __init__(self, resources, dispatchable, time_steps, values=None):
    self.resources = resources
    self.dispatchable = dispatchable
    self.time_steps = time_steps
    # TODO turn this into multiple DataArrays and a Dataset, and don't store 0s for resources that a component isn't involved with
    if values is None:
      values = np.zeros([len(time_steps), len(dispatchable), len(resources)])
    if isinstance(values, np.ndarray):
      coords = {'resource': resources, 'component': dispatchable, 'time': time_steps}
      self._record = xr.DataArray(values, coords=coords, dims=('time', 'component', 'resource'), name='dispatch')
    elif isinstance(values, xr.DataArray):
      self._record = values
    else:
      raise NotImplementedError

  def __add__(self, other):
    assert isinstance(other, DispatchRecord)
    # TODO this is a big slowdown spot!
    # In some profiling tests, this was roughly 2/3 of the run time!
    # Can we speed it up any way?
    # -> all the stuff I can find on combining XArrays, this is the only out of (concat, merge, combine) that
    #    correctly adds together the parts without doing some giant looping strategy.
    # -> I also tried doing this with pandas dataframes; for 2 dimensions, pandas is faster, but for 3 dimensions,
    #    xarray was notably faster. Our dimensions are "time", "component", and "resource".
    #    -> could we get away without tracking time? This seems to go against the "opt window" strategy.
    data = xr.concat((self._record, other._record),'__dummy').sum(dim='__dummy')
    # Note the problem is the line above, not the line below.
    new = DispatchRecord(self.resources, self.dispatchable, self.time_steps, values=data)
    return new

  def __sub__(self, other):
    assert isinstance(other, DispatchRecord)
    data = xr.concat((self._record, -1.0*other._record), dim='_dummy').sum(dim='__dumy')
    new = DispatchRecord(self.resources, self.dispatchable, self.time_steps, values=data)
    return new

  def __call__(self):
    return self._record

  def __repr__(self):
    return self._record.__repr__()

  def get_components(self):
    return self._record.component.values

  def get_resources(self):
    return self._record.resource.values

  def get_times(self):
    return self._record.time.values

  def get_net_quantity(self, comp_names, initials, resources):
    """
      Get the net resource quantity after all the interactions in the record.
      NOTE that the record holds RATES, not QUANTITIES, so we have to integrate against dt.
      The reported values are AMOUNTS, not RATES.
      @ In, comp_names, list(str), list of component names for which to get resource quantities
      @ In, initials, dict(str: float), initial amount of resource in component (default 0)
      @ In, resources, dict(str: list(str)), map of resources to collect for each component
      @ Out, res, dict(str: float), final quantites by component name
    """
    # useful for stuff like storages specifically, not sure if it has other applications
    data = self._record
    # TODO flexible dt; for now we consider them all equal
    times = data['time'].values[:2]
    dt = times[1] - times[0]
    # NOTE: deltas have a flipped sign, since negative is consuming and positive is producing
    ## but we want incoming to be positive and outgoing to be negative for quantities
    net_deltas = (- data.loc[{'component':comp_names}] * dt).sum(dim=('time'))
    res = {}
    for c in comp_names:
      # TODO multiple resources per component??
      resource = resources[c]
      delta = float(net_deltas.loc[{'component': c, 'resource': resource}].drop(['component', 'resource']))
      res[c] = delta + initials.get(c, 0.0)
    return res

  def add(self, res, comp, time, amt):
    #print('DEBUGG adding dispatch to component! res: {} comp: {} time: {} amt: {}'.format(res, comp, time, amt))
    #print('     starting amount:', float(self._record.loc[{'time': time, 'resource': res, 'component': comp}]))
    self._record.loc[{'time': time, 'resource': res, 'component': comp}] += amt
    #print('     ending amount:', float(self._record.loc[{'time': time, 'resource': res, 'component': comp}]))

  def set(self, resource, component, time_step, value):
    assert not np.isnan(value)
    self._record.loc[{'time': time_step, 'resource': resource, 'component': component.name}] = value

  def deconstruct(self, comp_map):
    d = {}
    name_template = 'Dispatch__{}__{}'
    for comp, res_list in comp_map.items():
      for res in res_list:
        name = name_template.format(comp, res)
        d[name] = self._record.loc[{'component':comp, 'resource':res}].drop(['component','resource']).values
    return d

  def get_comp_dispatch(self, time, comp):
    loc = {'time': time, 'component': comp}
    return self._record.loc[loc].drop(['time', 'component'])

  def print_component(self, component):
    print(self._record.xs(component.name, level='comp'))

  def prettyprint(self, string=False, drop=True):
    df = self._record.to_dataframe('Dispatch')
    if drop:
      df = df[df!=0].dropna()
    if string:
      return df.to_string()
    else:
      print(df)

  def time_slice(self, start, end):
    return self._record[start:end]

  def make_balance(self, additional=None):
    """ make a balance out of this dispatch """
    # balances are for resources, so sum over components
    df = self._record.sum(['time', 'component'])
    return df.to_series()




# independent methods

def copy_meta(meta):
  egret = meta.pop('EGRET', None)
  new = copy.deepcopy(meta)
  if egret is not None:
    meta['EGRET'] = egret
    new['EGRET'] = egret
  return new

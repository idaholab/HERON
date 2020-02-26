from __future__ import unicode_literals, print_function

import os
import sys
import copy
import numpy as np
import pandas as pd
import dill as pk
from collections import defaultdict
import xml.etree.ElementTree as ET
#import time as time_mod
import matplotlib.pyplot as mp
import pandas as pd
import time

import pprint
pp = pprint.PrettyPrinter(indent=4)


# load EGRET environment, so we can unpickle objects (why do I have to do this even with dill?)
## TODO how to know where it is?
#sys.path.append(os.path.expanduser('/Users/talbpw/projects/heron/src'))
sys.path.append(os.path.expanduser('/Users/gaira/heron/src'))
from DispatchScenario import Scenario, DispatchRecord
# from Economics import CashFlowInputWriter
# load RAVEN environment so we can use graph structure
## TODO dynamic path load
## required for unpickling
#raven_path = os.path.abspath(os.path.expanduser('/Users/talbpw/projects/heron/raven/framework'))
raven_path = os.path.abspath(os.path.expanduser('/Users/gaira/raven/framework'))
sys.path.append(raven_path)

# also need any functions set up from starting directory (up quite a few)
sys.path.append(os.path.join(os.getcwd(), '../../../'))

# cashflow tools
sys.path.append(os.path.join(raven_path,'..','plugins/CashFlow'))
#sys.path.append('/Users/gaira/Desktop/myHeron/egret/raven/scripts') # modify this to your working raven scripts folder

#sys.path.append(os.path.expanduser('/Users/talbpw/projects/heron')) #/CashFlow/src'))
sys.path.append(os.path.expanduser('/Users/gaira/heron/')) #/CashFlow/src'))
from CashFlow.src import CashFlows
from CashFlow.src.main import get_project_length
from CashFlow.src.main import run as CashFlow_run

# memory profiling swap option hack for kernprof
try:
  profile
except NameError:
  profile = lambda x: x


MAX_YEARS = 1e200
MAX_TIMES = 1e200
ONLY_DISPATCH = False

#######################
# VERBOSITY AND COLOR #
#######################
# global verbosity level
verbosity = 1# print, binary 0 basic, 1 everything -> based on recursion level

class ANSI_colors:
  RESET = '\033[0m' # reset color
  COMP = '\033[93m' # yellow
  RES = '\033[92m' # green
  TIME = '\033[96m' # cyan
  SCN = '\033[91m' # red

class NO_colors:
  RESET = '' # reset color
  COMP = ''  # yellow
  RES = ''   # green
  TIME = ''  # cyan
  SCN = ''   # red

# use colors
#colors = ANSI_colors
# don't use colors
colors = NO_colors
verb_kinds = ['econ', 'dispatch']


#################
# CUSTOM ERRORS #
#################
class NoScenarioError(Exception):
  """ for when there is no possible scenario to meet the request """
  pass
class ComponentAtCapacityError(Exception):
  """ for when a component is tapped out """
  pass

########
# MAIN #
########
def run(raven, inputs):
  #pk.dump(raven.e_price, open('DEBUGG_e_price.pk', 'wb'))
  #print('Dumped price to', 'DEBUGG_e_price.pk')


  for attr in ['MAX_TIMES', 'MAX_YEARS', 'ONLY_DISPATCH', 'verbosity']:
    val = getattr(raven, 'disp_'+attr, None)
    if val:
      setattr(sys.modules[__name__], attr, val[0])
  val = getattr(raven, 'disp_colors', None)
  if val:
    setattr(sys.modules[__name__], 'colors', ANSI_colors if val[0] else NO_colors)

  # for debugging
  #pk.dump((raven, inputs), open('rav_disp_extmod_args.pk', 'wb'))
  meta = {'EGRET':{}} # metadata information used during dispatch

  # load EGRET library --> todo pass the name +location in from outer through inner?
  found = False
  while not found:
    counter = 0
    try:
      with open('../egret.lib', 'rb') as lib:
        
        egret_case, egret_components, egret_sources = pk.load(lib)
        
      found = True
    except FileNotFoundError:
      print('WARNING: "egret.lib" not yet found; waiting to see if it shows up ...')
      counter += 1
    if counter > 6:
      look_for_me = os.path.join(os.path.getcwd(), '..', 'egret.lib')
      raise RuntimeError('Required library file "egret.lib" was not found at: "{}"'.format(look_for_me))


  meta['EGRET']['increments'] = egret_case.get_increments()
  #print("THESE ARE THE SOURCES",egret_sources)

  ###ADDED ONE MORE LINE########
  if egret_case.get_Resample_T() == None:
    pass
  else:
  
   
    meta['EGRET']['Resample_T']=egret_case.get_Resample_T()
  
   
  ##############################

  for comp in egret_components:
    # upgrade comps with capacities
    cap = inputs.get('{}_capacity'.format(comp.name), None)
    # TODO does this ever set a capacity when we really didn't want to?
    if cap is not None:
      if check_verb(1, 'dispatch'):
        print('RUN: setting comp "{}" cap to {}'.format(comp.name, float(cap)))
      comp.set_capacity(cap)
    if check_verb(1, 'dispatch'):
      print('Component "{}" summary:'.format(comp.name))
      comp.print_me()

  # construct graph
  ## FIXME move this to EGRET and pass in lib, so we can check before writing and running inputs?
  ## note that we can't send ALL the components to the graph,
  resource_map = construct_graph(egret_components)
  meta['EGRET']['resource_map'] = resource_map
  meta['EGRET']['econ_evaluation'] = 'INCREMENTAL'

  # do the dispatch
  ## get the project years
  econ_global_settings, econ_comps, project_life = _build_economics_objects(egret_case, egret_components)
  meta['EGRET']['project_life'] = project_life
  # GET YEARS from the cluster information, not from the project life!
  cluster_info = _obtain_cluster_details(egret_case, egret_sources)

  # store ARMA data
  pivot_id = 'Time' # TODO hardcoded pivot!
  new_indices = {}
  for var, indices in getattr(raven, '_indexMap', [{}])[0].items():
    if pivot_id in indices:
     
      
      ###############################INTERPOLATION#############################
      Res=egret_case.get_Resample_T()
      Res2=egret_case.get_hist_length()
      resample=int(Res/Res2)
    
      
      
      
      idx=inputs[indices[0]]
      old_dt=idx[-1]-idx[-2]
      new_dt=idx[-1]/float(Res)
     
      if Res!= None:
        temp = getattr(raven, var)
        Array=np.empty((Res,np.shape(getattr(raven,var))[1]))
        for i in range(0,np.shape(getattr(raven,var))[1]):
          Temp=[]
          for j in range(1,len(inputs[indices[0]])):
           
            temporary_arma=egret_sources[0].interpolation(np.array([idx[j-1],idx[j]]),(temp[j-1:j+1,i]).squeeze())
            
        
            new_time_array = np.linspace(idx[j-1],idx[j],(idx[j]-idx[j-1])/new_dt+1)
         
            interpolated_signal = temporary_arma(new_time_array)
            Temp.append(list(interpolated_signal[0:resample]))
          
           
            
          Temp=np.hstack(Temp)
          #Temp=Temp[0::3]
          
          Array[:,i]=np.array(Temp)
          
      setattr(raven,var,Array)
      ########################DONE#############################################
      new, dims = reshape_variable_as_clustered(var, getattr(raven, var), raven._indexMap[0], cluster_info, pivot_id)
      new_name = var + '_cluster_shaped'

      set_raven_var(raven, new_name, new)
      new_indices[new_name] = dims
  if new_indices:
    raven._indexMap[0].update(new_indices)


  # get years we're running

  years = range(cluster_info['macro_first'], cluster_info['macro_last']+1)
  temporary_var=(meta['EGRET'].keys())
 
  

 
  dispatches = dispatch_all_years(years, cluster_info, inputs, egret_case, egret_sources, egret_components, meta)

  # add some other things asked for to the collection
  time = raven.Time # TODO flexible name?
  set_raven_var(raven, 'time_delta', time[1]-time[0]) # TODO always consistent?

  # FIXME do cashflow for cluster dispatches, then expand with multipliers for yearly entries.
  results = save_dispatch_vars(raven, years, egret_components, dispatches)

  if ONLY_DISPATCH:
    #years = getattr(raven, 'Year', None)
    #if years:
    #  years = np.arange(years[0], years[0] + MAX_YEARS + 1)
    #  setattr(raven, 'Year', years)
    set_raven_var(raven, 'NPV', 0.0)
    print('DEBUGG breaking to debug dispatch only!')
    return
  # DEBUGG for fast running, pickle all this crap so we can test it out
  #pk.dump((cluster_info, egret_case, egret_components, dispatches, results, meta, inputs), open('restart_cashflow.pk', 'wb'))
  # run cashflow
  meta['EGRET']['econ_evaluation'] = 'FINAL'
  metrics = run_cashflow(cluster_info, egret_case, egret_components, dispatches, results, meta, inputs)
  # save cashflow metrics as results
  for metric, value in metrics.items():
    set_raven_var(raven, metric, value)


  #raise NotImplementedError('SUCCESSFULLY FINISHED INNER, but crashing this here so I can debug some stuff. - talbpw')

def set_raven_var(raven, name, var):
  print('Setting RAVEN variable "{}" with shape {}'.format(name, np.atleast_1d(var).shape)) # TODO remove atleast1d for speed
  setattr(raven, name, var)

# MOVE ME!
def _obtain_cluster_details(case, sources):
  working_dir = case.get_working_dir('inner')
  info = {}
  for source in sources:
    if source.is_type('ARMA'):
      name = source.name
      # open the rom meta file
      meta_fname = '{}_meta.xml'.format(name) # TODO this is tied to the lcoe_sweep_opt.py naming template!
      if os.path.isfile(meta_fname): # TODO this should only be false when scripting!
        to_open = meta_fname
      else:
        to_open = os.path.join(os.path.abspath(working_dir), meta_fname)
      meta = ET.parse(to_open).getroot()
      # read macro global parameters
      ## TODO other kinds of ROMs?
      info_node = meta.find(name).find('InterpolatedMultiyearROM')
      macro_param = info_node.find('MacroParameterID').text.strip()
      info['macro_param'] = macro_param
      info['num_macro_steps'] = int(info_node.find('MacroSteps').text)
      info['macro_first'] = int(info_node.find('MacroFirstStep').text)
      info['macro_last'] = int(info_node.find('MacroLastStep').text)
      info['clusters'] = {}
      # read each macro step parameters (primarily clustering info)
      for macro_node in meta.find(name).findall('MacroStepROM'):
        step_num = int(macro_node.attrib[macro_param])
        info['clusters'][step_num] = {}
        # we don't need the global info for the macro step, just get the clusters
        for cluster_node in macro_node.findall('ClusterROM'):
          cluster_id = int(cluster_node.attrib['cluster'])
          num_represent = len(cluster_node.find('segments_represented').text.split(','))
          indices = list(int(x) for x in cluster_node.find('indices').text.split(','))
          info['clusters'][step_num][cluster_id] = {'num_represent': num_represent,
                                                    'indices': indices}
  return info

def dispatch_all_years(years, cluster_info, inputs, egret_case, egret_sources, egret_components, meta):
  
  # find out if we're clustering ARMAs (FIXME we always are?); if so, find out the details
  # cluster_details has info BY MACRO STEP for the clusters of that year.
  # main loop
  dispatches = []
  for y, year in enumerate(years):

    if y + 1 > MAX_YEARS:
      # TODO create fake info!

      break
    print('*   *   *   *   *   *   *   *')
    print('*                            ')
    print('*   Starting year "{}"       '.format(year))
    print('*                            ')
    print('*   *   *   *   *   *   *   *')
    meta['EGRET']['sim_year'] = year
    meta['EGRET']['sim_year_index'] = y
    # This is where the magic happens, dispatch the year!
    ## TODO clustered years?? Should be done piecemeal!

    dispatch_map, meta = dispatch_clusters(inputs, egret_case, egret_sources, egret_components, meta, cluster_info)
    # for a first stab, expand cluster dispatches into representative yearly performance
    dispatches.append(dispatch_map)
  return dispatches

def save_dispatch_vars(raven, years, egret_components, dispatches):
  clustered = True
  indexMap = getattr(raven, '_indexMap', [{}])[0]
  results = defaultdict(list)
  comp_map = dict((comp.name, comp.get_inputs().union(comp.get_outputs())) for comp in egret_components)
  if clustered:
    clusters = sorted(list(dispatches[0].keys())) # TODO assuming same clusters every year; we aren't crazy.
    # final results are an ND array with intended dimensions (cluster, year, time)
    ## to achieve this, stack the years together, then stack the clusters together
    cluster_results = defaultdict(list)
    for cluster in clusters:
      # get the yearly activities for this cluster and stack them together in year_results
      year_results = defaultdict(list)
      for y, cluster_dispatches in enumerate(dispatches):
        # get the dispatch for this year, for this cluster
        dispatch = cluster_dispatches[cluster]['dispatch']
        # deconstruct the activities (format: activity = <component>__<resource>: <float>)
        var_dict = dispatch.deconstruct(comp_map)
        # stack the activities for this cluster for this year into the year stack
        for k, v in var_dict.items():
          year_results[k].append(v)
      for y_nope in range(y+1, len(getattr(raven, 'Year'))):
        for k, v in var_dict.items():
          year_results[k].append(v)
      # combine the year stack (list(1darray)) into activity for this cluster (2darray)
      for activity, store in year_results.items():
        cluster_results[activity].append(np.asarray(store))
        # dimensions for each of these activities are now [year, seconds]
    # combine the cluster stack (list(2darray)) into total activity (3darray)
    for activity, store in cluster_results.items():
      results[activity] = np.asarray(store)
      # dimensions for each activity is now [cluster, year, seconds]
    nd_dims = np.array(['Cluster', 'Year', 'ClusterTime'])
    # store the Cluster dimension
    results['Cluster'] = np.array(clusters)
    set_raven_var(raven, 'Cluster', results['Cluster'])
    # reset the Time dimension, since we're clustering.
    ## Lots of potential for problems here if the clusters aren't the same length!
    results['ClusterTime'] = dispatch.get_times()
    set_raven_var(raven, 'ClusterTime', results['ClusterTime'])

  else: # no clusters case; probably in disrepair
    # break up dispatch into RAVEN-acceptable variables -> TODO how for clusters??
    results = defaultdict(list)
    for y, dispatch in enumerate(dispatches):
      var_dict = dispatch.deconstruct(comp_map)
      for k, v in var_dict.items():
        results[k].append(v)
    for activity, store in results.items():
      results[activity] = np.asarray(store)
    # dimensions for each of these activities are now [year, seconds] -> add cluster?
    nd_dims = np.array(['Year', 'Time']) # TODO is it always "Time"?
  # end if-clustered

  # store the Year dimension
  results['Year'] = np.atleast_1d(years[:len(dispatches)]) # len(dispatches) instead of years in case MAX YEARS
  set_raven_var(raven, 'Year', results['Year'])

  # update the mapping for variables in results
  for variable, value in results.items():
    if variable not in ['ClusterTime', 'Year', 'Cluster']:
      set_raven_var(raven, variable, value)
      indexMap[variable] = nd_dims

  # keep the index map as well
  results['_indexMap'] = indexMap
  set_raven_var(raven, '_indexMap', indexMap)

  return results

################
# MAIN DRIVERS #
################
def dispatch_clusters(full_inputs, case, sources, components, meta, all_cluster_info):
  """ dispatch the clusters individually """
  year = meta['EGRET']['sim_year']
  clusters_info = all_cluster_info['clusters'][year]
  indexMap = full_inputs.get('_indexMap', [{}])[0]
  cluster_dispatches = {}
  for c, cluster_info in clusters_info.items():
    print('DISPATCH: solving cluster', c)
    #time_mod.sleep(2000)
    inputs, start, end = select_cluster_data_from_history(full_inputs, indexMap, cluster_info, 'Time', exceptions=['Year'])
    meta['EGRET']['cluster_selector'] = slice(start, end)
    # TODO right now it looks like nothing in the meta depends on the pivot, so skipping
    # dispatch the cluster
    # TODO meta should be deepcopied or carried??
    dispatch, meta = dispatch_one_year(inputs, case, sources, components, meta)
    print('FINAL dispatch, year {} cluster {}:'.format(year, c))
    recdfpr(-1, dispatch.prettyprint(string=True, drop=True))
    cluster_dispatches[c] = {'dispatch': dispatch,
                             'multiplicity': cluster_info['num_represent'],
                             'start': start,
                             'end': end}

  return cluster_dispatches, meta

def dispatch_one_year(inputs, case, sources, components, meta):
  """ Master dispatcher for components """
  # TODO move much of this to the dispatch_all_years!
  time_steps = inputs['Time']
  print("These are the time steps",time_steps)
  tttt
  # who's dispatchable?
  disp_fixed = []    # components that are forced; can't be dispatched
  disp_depend = []   # components that can be dispatched, but shouldn't be primary source of perturbations
  disp_indep = []    # components that can be dispatched, and ARE the primary source of perturbations
  resources = set()  # all resources tracked in this analysis
  storages = []
  for comp in components:
    resources.update(comp.get_inputs())
    resources.update(comp.get_outputs())
    interaction = comp.get_interaction()
    if interaction.is_dispatchable() == 'fixed':
      disp_fixed.append(comp)
    elif interaction.is_dispatchable() == 'dependent':
      disp_depend.append(comp)
    else: #independent
      disp_indep.append(comp)
    # if a storage, we need to track it so we can track levels
    if interaction.is_type('Storage'):
      storages.append(comp)

  # create storage structure
  ## need dispatch to depend on both resource and component -> FIXME could this be sparse though? XArray?
  resources = sorted(list(resources))
  disp_comp_names = list(c.name for c in disp_indep + disp_depend + disp_fixed)
  dispatch = DispatchRecord(resources, disp_comp_names, time_steps)

  # the following do not change from time window to time window
  state = {'components': {'independent': disp_indep,
                          'dependent': disp_depend,
                          'fixed': disp_fixed,
                          'storages': storages},
           'resource map': meta['EGRET']['resource_map'],
           'raven vars': inputs}
  window_size = 1 # turns out in 2019 we don't need to dispatch any windows, maybe? Let's leave possibility in though.
  for t, time in enumerate(time_steps):
    if t + window_size >= len(time_steps):
      break
    if t >= MAX_TIMES:
      break # DEBUGG for timing purposes
    meta['EGRET']['time_window'] = time_steps[t:t + window_size]
    time_window = time_steps[t: t+window_size]
    # build base scenario for this window
    state['dispatch'] = dispatch
    state['meta'] = meta
    state['time window'] = time_window
    state['initial step number'] = t
    scenario = Scenario('base', state)
    scenario = dispatch_window(scenario, case, window_size=window_size)
    # set windowed portions from new dispatch to original (not addition!)
    target = {'time': time_window}
    dispatch().loc[target] = scenario.get_dispatch('original')().loc[target] #we can use original because we accept delta in dispatch_window at end
    meta = scenario.get_meta(dispatch='original', make_copy=True)
    # TODO does some part of the meta need clearing as the window rolls forward? User option?
  # done with dispatching!
  return dispatch, meta

def dispatch_window(scenario, case, window_size=1):
  #inputs, dispatch, t, time, case, sources, meta, forced, dependent, dispatchable, storages, resources, window_size=1):
  """
    Dispatches components in a fixed window given the case, undispatchable components, etc
  """
  start = scenario.get_start_time()
  end = scenario.get_end_time()
  print('='*100)
  print('Dispatching time window {} through {}'.format(clr(start, 'TIME'), clr(end, 'TIME')))
  print('='*100)
  # TODO start scenario HERE, not per-time-step! -> or start it at birth, back before dispatch_window is called?
  # window_balance = BalanceRecord(scenario.resources, time)
  # dispatch all fixed components at full production
  scenario = _dispatch_fixed(scenario)
  #scenario.accept_delta()
  print('DISPATCH: status after dispatching fixed units:')
  print_scenario(-1, scenario)
  # dispatch minima --> should only happen on the FIRST time step, otherwise we use last state as starting point!
  ## note that any limitations should be handled by the component during the call
  #if t == 0:
  #  # TODO loop through all the time steps to set them all the first time through!
  #  ## -> for now, we do window size 1, so this isn't a big deal.
  #  _, dispatch, meta = _dispatch_minima(window_balance, dispatch, t, window_size, meta, inputs, dependent+dispatchable)
  #else:
  scenario = _dispatch_minima(scenario) #window_balance, dispatch, t, window_size, meta, inputs, dependent+dispatchable)
  #scenario.accept_delta()
  print('DISPATCH: status after dispatching unit minima:')
  print_scenario(-1, scenario)
  #print_balance(t, window_size, time, window_balance, dispatch)
  # now we've dispatched the baseline from all components, so proceed to optimally dispatch the units.
  ## create baseline scenario
  scenario = _dispatch_to_zero(scenario)
  scenario.accept_delta()
  print('DISPATCH: status after dispatching to zero:')
  print_scenario(-1, scenario)
  # OLD window_balance, dispatch, t, time, window_size, meta, inputs, dispatchable, dependent, forced, storages, case)
  ## NOTE this will have to be modified slightly if the case metric is LCoE instead of NPV.
  if case._metric != 'NPV':
    raise NotImplementedError('Currently only NPV maximization search has been implemented; some additional work is needed for others.')
  recpr(0, 'Successfully dispatched to ZERO. Status:')
  print_scenario(0, scenario)
  # now try to be profitable
  scenario = _dispatch_profit(scenario)
  scenario.accept_delta()
  recpr(0, 'Successfully searched for profit. Status:')
  print_scenario(0, scenario)
  return scenario

def select_cluster_data_from_history(variables, indexMap, cluster_info, pivot, exceptions=None):
  """
    Given clustering information, truncate variables in a dict to be the desired cluster only.
    Assumes clusters are given in a contiguous history, which is kind of a crappy way of doing it tbh.
    @ In, variables, dict, {names: np.ndarray(vals)} to convert from "full" histories to cluster parts
    @ In, indexMap, dict, {var: [dims]} mapping to explain dimentionality
    @ In, cluster_info, dict, information about the cluster properties
    @ In, pivot, string, name of the pivot parameter (the one that's clustered)
    @ In, exceptions, list(str), optional, variables (including other dimensions) to ignore
    @ Out, new, dict, modified dict similar to "variables" but truncated to cluster
  """
  if exceptions is None:
    exceptions = []
  start, end = cluster_info['indices']
  selector = slice(start, end)
  # n_times = len(variables.get('Time', []))
  # n_ctimes = len(variables.get('ClusterTime', []))
  new = {}
  for var, vals in variables.items():
    dims = indexMap.get(var, None)
    if dims is None:
      if isinstance(vals, np.ndarray):
        if vals.size > 1:
          if var == pivot:
            new[var] = vals[selector]
          elif var in exceptions:
            new[var] = vals
          else:
            print('DEBUGG what is this?', var, vals)
            raise NotImplementedError # what is this?
        else:
          # these are 1-d arrays of floats (unless pivot len is 1, which would be weird)
          new[var] = vals
      else:
        print('DEBUGG 是什么?', var, vals)
        raise NotImplementedError # 是什么？
    else:
      # sometimes the variable is already set to ClusterYear, so no mods needed
      ## but if it depends on Time, we need to chop it up
      if 'Time' in dims:
        # we need to truncate!
        pivotIndex = list(dims).index(pivot)
        nd_selector = [None] * len(dims)
        nd_selector[pivotIndex] = selector
        new[var] = vals[selector]
      elif 'ClusterTime' in dims:
        new[var] = vals
      else:
        print('DEBUGG que es eso?', var, vals)
        raise NotImplementedError # que es eso?
  return new, start, end

def reshape_variable_as_clustered(var, values, indexMap, cluster_info, pivot):
  macro_first = cluster_info['macro_first']
  macro_last = cluster_info['macro_last']
  num_macro = cluster_info['num_macro_steps']
  macro_id = cluster_info['macro_param']

  num_cluster = len(cluster_info['clusters'][macro_first])
  num_pivot = cluster_info['clusters'][macro_first][num_cluster - 1]['indices'][1]
  # TODO assuming the lengths are the same for each cluster!
  cl_pivot = num_pivot - cluster_info['clusters'][macro_first][num_cluster - 1]['indices'][0]
  new = np.zeros((cl_pivot, num_macro, num_cluster))
  #new = np.zeros((num_cluster, num_macro, cl_pivot))
  pivot_index = indexMap[var].index(pivot)
  for cluster in range(num_cluster):
    #time.sleep(2000)
    start, end = cluster_info['clusters'][macro_first][cluster]['indices']
    selector = [None, None]
    selector[pivot_index] = slice(start, end)
    data = values[selector].squeeze()
    new[:, :, cluster] = data
  dims = ['ClusterTime', macro_id, 'Cluster'] # TODO hardcoded clustertime, cluster
  return new, dims

#############
# ECONOMICS #
#############
@profile
def run_cashflow(cluster_info, case, heron_components, dispatches, results, meta, inputs):
  results.update(inputs) # TODO is this right for clustering???
  clustered = True
  if clustered:
    # build final cashflow objects, for storing results
    final_settings, final_components, project_life = _build_economics_objects(case, heron_components)
    indexMap = results.get('_indexMap', [{}])[0]
    # for each year, for each cluster, figure out the economic impacts
    ## then translate those back to the main, final cashflow objects
    years = list(range(cluster_info['macro_first'], cluster_info['macro_last']+1))
    for y, cluster_dispatches in enumerate(dispatches):
      print('DEBUGG YEARS:', years)
      print('DEBUGG YEAR:', y, years[y])
      year = years[y]
      for cluster, info in cluster_info['clusters'][year].items():
        print('CASHFLOW SETUP -> YEAR {} CLUSTER {}'.format(year, cluster))
        # get the dispatch for this cluster, this year
        dispatch = cluster_dispatches[cluster]['dispatch']
        # how many times in this year was this cluster representing?
        multiplicity = info['num_represent']
        # build objects
        _, local_comps, _ = _build_economics_objects(case, heron_components)
        print(' ... cashflow setup: economics objects built')
        cluster_results, start, end = select_cluster_data_from_history(results, indexMap, info, 'Time', exceptions=['Cluster', 'ClusterTime', 'Year'])
        print(' ... cashflow setup: cluster data selected')
        values_dict = {'raven_vars': cluster_results, 'meta': meta}
        # TODO user values in meta might be all wacky and not cluster-focused at this point!
        ## FIXME might need to keep meta seperate by cluster.
        meta['EGRET']['sim_year'] = year
        meta['EGRET']['sim_year_index'] = y
        meta['EGRET']['cluster_selector'] = slice(start, end)
        meta['EGRET']['dispatch'] = dispatch
        # TODO need time, time window?
        #print('DEBUGG CASHFLOW META:')
        #pprint.pprint(meta['EGRET'])
        # setup intrayear cashflow
        setup_intrayear_cashflows(y, dispatch, heron_components, local_comps, values_dict)
        print(' ... cashflow setup: year cashflows calculated')
        # extract data into final objects
        for c, heron_comp in enumerate(heron_components):
          comp_name = heron_comp.name
          heron_comp = heron_comp.get_economics()
          comp = local_comps[comp_name]
          #for comp_name, comp in local_comps.items():
          final_comp = final_components[comp_name]
          #heron_comp = heron_components[comp_name]
          assert comp.name == final_comp.name
          final_cashflows = final_comp.get_cashflows()
          for f, cf in enumerate(comp.get_cashflows()):
            final_cf = final_cashflows[f]
            heron_cf = heron_comp.get_cashflows()[f]
            assert cf.name == final_cf.name == heron_cf.name
            if cf.type == 'Recurring':
              # hourly period need to be multiplied by the weeks represented by the cluster
              if heron_cf.get_period() == 'hour': # need to determine the period of this cf! Only the corresponding heron cf knows.
                # extract the yearly data
                #cf.compute_intrayear_cashflow(y, cf._yearly_cashflow, 1.0)
                print('DEBUGG yearly cf:', cf._yearly_cashflow, y)
                year_cf = cf._yearly_cashflow[y]
                final_cf._yearly_cashflow[y+1] += year_cf
              # yearly periods only need be calculated once per yer, and without cluster multiplicity
              elif heron_cf.get_period() == 'year':
                if cluster == 0:
                  final_cf._yearly_cashflow[y+1] += cf._yearly_cashflow[y+1]
              # otherwise ... shouldn't get here.
              else:
                raise NotImplementedError # 这个真么样？
            elif cf.type == 'Capex':
              if y == cluster == 0:
                final_comp._cash_flows[f] = cf # TODO is this right? Should only need one total for each capex ...
            else:
              raise NotImplementedError
        print(' ... cashflow setup: yearly cluster values extracted')
    settings = final_settings
    cf_comps = final_components
    print('Creating amortization schemes ...')
    create_amortizations(final_components)
  else: # probably in disrepair.
    # set up the cashflows (put floats in all the right places) so that CashFlow can take it away.
    # ? can we make a set of cf_comps for each cluster, then combine their alphas?
    ## -> why the heck not?
    settings, cf_comps = setup_cashflows(case, heron_components, dispatches, results, meta, inputs)

  # CashFlow, take it away.
  print('****************************************')
  print('* Starting final cashflow calculations *')
  print('****************************************')
  cf_metrics = CashFlow_run(settings, list(cf_comps.values()), inputs)

  print('DEBUGG final cashflow metrics:')
  for k, v in cf_metrics.items():
    print('  ', k, v)
  return cf_metrics

def setup_cashflows(case, heron_components, dispatches, results, meta, inputs):
  # In order to run CashFlows.main.run, we need 3 things:
  #  - settings, the global information about the cash flows (indicators, wacc, etc)
  #  - components, the (economic) components, which are basically grouped up cashflows
  #  - variables, the pool of variables and values that include all the drivers needed

  # TODO maybe this should get run YEARLY during the dispatch, instead of all at the end?
  results.update(inputs)
  # 1. Build the global settings
  # create CashFlow global settings
  cf_settings, cf_components, project_life = _build_economics_objects(case, heron_components)
  # TODO debug verbosity settings?
  cf_settings._verbosity = 0
  ## cf_settings are the settings needed for the cash flow run
  ## raw_components need to be converted into CashFlow components still

  # 2. Build the components
  # start from the raw_components (egret version of cashflow components) and make them into real CashFlow ones
  #cf_components = {}
  # build variables dictionary for cashflow evaluations
  values_dict = {'raven_vars': results, 'meta': meta} # TODO deepcopy??

  #print('DEBUGG converting comp "{}" ({}) to cash flow ...'.format(cfg._component.name, type(cfg)))
  print('Converting cash flows from EGRET to CashFlow objects ...')
  for y, dispatch in enumerate(dispatches):
    setup_intrayear_cashflows(y, dispatch, heron_components, cf_components, values_dict)
  # END loop over years
  # now implement any amortizations
  print('Creating amortization schemes ...')
  create_amortizations(cf_components)

  return cf_settings, cf_components

def _build_economics_objects(heron_case, heron_components):
  """ creates CashFlow.Case initialization parameters and gets the economic portion of components """
  # get the economics information from each component
  heron_econ_comps = list(comp.get_economics() for comp in heron_components)
  # build the economics settings for CashFlow
  econ_global_params = heron_case.get_econ(heron_econ_comps)
  econ_global_settings = CashFlows.GlobalSettings()
  econ_global_settings.set_params(econ_global_params)
  econ_global_settings._verbosity = 0 # TODO make a user option!
  # build the components for CashFlow
  cf_components = {}
  for c, cfg in enumerate(heron_econ_comps):
    # cfg is the cashflowgroup connected to the heron component
    # get the heron component we're attached to
    heron_comp = heron_components[c]
    comp_name = heron_comp.name
    # build the CashFlow equivalent
    cf_comp = CashFlows.Component()
    cf_comp_params = {'name': comp_name,
                      'Life_time': cfg.get_lifetime(),
                      # StartTime, Repetitions, tax, inflation
                     }
    cf_comp.set_params(cf_comp_params)
    cf_components[comp_name] = cf_comp
  # pause to calculate the project life
  project_life = get_project_length(econ_global_settings, list(cf_components.values()))

  # create all the CashFlow.CashFlows (cf_cf) for the CashFlow.Component
  for c, cfg in enumerate(heron_econ_comps):
    heron_comp = heron_components[c]
    comp_name = heron_comp.name
    # find the CashFlow equivalent
    cf_comp = cf_components[comp_name]
    ## add them all at once for speed and simplicity
    cf_cfs = []
    for heron_cf in cfg.get_cashflows():
      cf_name = heron_cf.name
      # build is slightly different depending on the type. # TODO factory?
      if heron_cf._type == 'repeating':
        cf_cf = CashFlows.Recurring()
        cf_cf_params = {'name': cf_name,
                        'X': 1.0,
                        'mult_target': heron_cf._mult_target,
                       }
        cf_cf.set_params(cf_cf_params)
        cf_cf.init_params(project_life + 1)
        cf_cfs.append(cf_cf)
      elif heron_cf._type == 'one-time':
        cf_cf = CashFlows.Capex()
        cf_cf.name = cf_name
        cf_cf.init_params(cf_comp.get_lifetime())
        # alpha, driver for the specific year aren't known a priori, so set those later.
        cf_cfs.append(cf_cf)
      else:
        raise NotImplementedError # TODO other cashflow types ??
    cf_comp.add_cashflows(cf_cfs)
  return econ_global_settings, cf_components, project_life

def setup_intrayear_cashflows(y, dispatch, heron_comps, cf_comps, values_dict):
  """ sets up the drivers/prices for cash flows within a given year """
  times = dispatch.get_times()
  T = len(times)
  alphas = defaultdict(lambda: np.zeros(T))
  drivers = defaultdict(lambda: np.zeros(T))
  for t, time in enumerate(times):
    # DEBUGG!
    if t > MAX_TIMES:
      break
    for c, comp in enumerate(heron_comps):
      name = comp.name
      cf_comp = cf_comps[name]
      first_time = y == t == 0
      setup_comp_cashflows(t, time, comp, cf_comp, dispatch,
                           values_dict, alphas, drivers, first_time)
  for cf, alpha in alphas.items():
    driver = drivers[cf]
    print('DEBUGG doing year {} cluster ?? summary cashflow for'.format(y), cf.name)
    print('DEBUGG   alpha:', alpha)
    print('DEBUGG   driver:', driver)
    ### FIXME original
    # cf.compute_yearly_cashflow(alpha, driver)
    cf.compute_intrayear_cashflow(y, alpha, driver)
    print('DEBUGG   result:', cf._yearly_cashflow)
  #return alphas, drivers

def setup_comp_cashflows(t, time, heron_comp, cf_comp, dispatch, values_dict, alphas, drivers, first_time):
  """ sets up all the cashflows for a given component AT A GIVEN TIME t"""
  name = cf_comp.name
  activity = dispatch.get_comp_dispatch(time, name).to_series()
  values_dict['raven_vars'].update(activity.to_dict())
  for c, cashflow in enumerate(cf_comp.get_cashflows()):
    heron_cf = heron_comp.get_economics().get_cashflows()[c]
    assert heron_cf.name == cashflow.name
    if cashflow.type == 'Recurring':
      period = heron_cf.get_period()
      # for hourly recurring cashflows, calculate the contribution for this hour
      # for once-a-year recurring cashflows, only calculate this contribution once
      if period == 'hour' or (t == 0 and period == 'year'):
        alpha, driver = setup_cashflow_hourly_recurring(t, cashflow, heron_cf, values_dict)
      elif period == 'year':
        alpha, driver = 0, 0
      else:
        raise NotImplementedError('Got unrecognized Recurring "period" type of {}'.format(period))
      # store the alpha, driver values
      alphas[cashflow][t] = alpha
      drivers[cashflow][t] = driver
    elif cashflow.type == 'Capex':
      if first_time:
        setup_cashflow_onetime(t, cashflow, heron_cf, values_dict)
    else:
      raise NotImplementedError('Unrecognized cashflow type: "{}"'.format(cashflow.type))

def setup_cashflow_hourly_recurring(t, cf, heron_cf, values_dict):
  """ Sets up a recurring (sales) cash flow """
  assert cf.type == 'Recurring'
  heron_params = heron_cf.calculate_params(values_dict, times=[t])
  alpha = heron_params['alpha']
  driver = heron_params['driver']
  return alpha, driver

def setup_cashflow_onetime(t, cf, heron_cf, values_dict):
  """ Sets up a one-time (CAPEX) cashflow """
  heron_cf_params = heron_cf.calculate_params(values_dict, times=[t])
  cf_params = {'name': cf.name,
               # 'multiply': ,
               'mult_target': heron_cf._mult_target,
               'depreciate': heron_cf._depreciate,
               # TODO I don't really like these placeholders; what's the right way to do this?
               'alpha': heron_cf_params['alpha'],
               'driver': heron_cf_params['driver'],
               'reference': heron_cf_params['ref_driver'],
               'X': heron_cf_params['scaling'],
              }
  cf.set_params(cf_params)

def create_amortizations(comps):
  """ creates amortization/depreciation cashflows, based on capex """
  for name, comp in comps.items():
    for cf in comp.get_cashflows():
      if cf._depreciate is not None:
        cf.set_amortization('MACRS', cf._depreciate)
        amorts = comp._create_depreciation(cf)
        comp.add_cashflows(amorts)

###################
# STARTER METHODS #
###################
def construct_graph(components):
  """ construct dict of resources to what components either consume or produce that resource """
  res_info = {}
  # add components
  print('DEBUGG comps:', components)
  for comp in components:
    ins = comp.get_inputs()
    outs = comp.get_outputs()
    for res in ins:
      if res not in res_info:
        res_info[res] = {'produced by': [], 'consumed by': [comp]}
      else:
        res_info[res]['consumed by'].append(comp)
    for res in outs:
      if res not in res_info:
        res_info[res] = {'produced by': [comp], 'consumed by': []}
      else:
        res_info[res]['produced by'].append(comp)
  # add sources --> no, because they should be made available through a component

  # DEBUGG
  for res, info in res_info.items():
    print('RESOURCE "{}":'.format(res))
    print('    produced by:', list(x.name for x in info['produced by']))
    print('    consumed by:', list(x.name for x in info['consumed by']))
  # sanity checking
  for res, info in res_info.items():
    if not info['produced by']:
      raise IOError('Resource "{}" is not produced by any component or source!'.format(res))
    if not info['consumed by']:
      print('WARNING: resource "{}" is not consumed by any components!'.format(res))
  return res_info

def _dispatch_fixed(scenario):
  recpr(0, ' ... dispatching fixed components ...')
  for comp in scenario.fixed:
    for t, time in enumerate(scenario.get_time_steps()):
      step = scenario.get_start_step() + t
      balance, meta = comp.produce_max(scenario.get_meta(dispatch='original', make_copy=False),
                                       scenario.raven_vars,
                                       scenario.get_dispatch('original'),
                                       step)
      scenario.dispatch(balance, comp, step)
  recpr(0, ' ... fixed components have been dispatched!')
  return scenario

def _dispatch_minima(scenario): # step_balance, dispatch, t, window_size, meta, inputs, comps):
  recpr(0, ' ... dispatching component minima ...')
  for comp in scenario.dispatchable + scenario.dependent:
    for t, time in enumerate(scenario.get_time_steps()):
      step = scenario.get_start_step() + t
      balance, meta = comp.produce_min(scenario.get_meta(dispatch='original', make_copy=False),
                                       scenario.raven_vars,
                                       scenario.get_dispatch('original'),
                                       step)
      scenario.dispatch(balance, comp, step)
      #if any(list(balance.values())):
      #  update_comp_dispatch(dispatch, comp, balance, step)
      #  update_resource_balance(balance, step_balance, window_time_step)
  recpr(0, ' ... component minima have been dispatched!')
  return scenario

#######################
# PARTIAL DISPATCHERS #
#######################
def _dispatch_to_zero(scenario):
  """ dispatch components (based on increments) to get the balance to 0 at each time step within the window """
  recpr(0, ' ... dispatching components to meet base requirements ...')

  # resource_map = scenario.resource_map
  # TODO should we really be looping time steps here?
  assert len(scenario.get_time_steps()) == 1
  for t, time in enumerate(scenario.get_time_steps()):
    step = scenario.get_start_step() + t
    #for s, step in enumerate(range(t, t+window_size)):
    print('-'*100)
    print('Starting step balancing for step {}: {}'.format(clr(step, 'TIME'), clr(time, 'TIME')))
    # get the balance for this time step
    # step_balance = window_balance.get_slice(step)
    # create new BASELINE scenario for this step
    ## initialize with all 0s for balance, but perturb with minima for step
    # base_balance = BalanceRecord(step_balance.resources, step_balance.time_steps)
    # scenario = Scenario('base', base_balance, dispatch, storages, resource_map, meta, inputs, dispatchable, dependent, time, t)
    # perturb values according to step balance dispatch
    # scenario.set_perturbation(step_balance)
    # scenario = Scenario(step_balance, dispatch, resource_map, meta, inputs, dispatchable, dependent, time, t)
    try:
      scenario = _balance_resources_timestep(scenario)
    except NoScenarioError:
      print('\n\n\nUnfortunately, no solution was found for the given system.\n\n')
      raise NoScenarioError
    print('-'*100)
    print('Step {}: {} has been dispatched to meet requirements.'.format(clr(step, 'TIME'), clr(time, 'TIME')))
    print('-'*100)
  return scenario

def _dispatch_profit(scenario):
  """ past dispatching to zero, attempt to dispatch profitably """
  # TODO time-dependent???
  # get nominal cost of scenario
  recpr(0, 'Calculating nominal scenario cost ...')
  nominal_profit = _get_scenario_cost(scenario, 0)
  increments = scenario.get_meta(dispatch=None, make_copy=False)['EGRET']['increments']
  finished = False
  while not finished:
    # find out which component could be the most profitable next
    best_option = {'scenario': None, 'profit': None}
    for comp in scenario.dispatchable:
      new_scenario = None
      # make a scenario branch
      new_scenario = scenario.add_branch(comp.name, pin=[comp])
      # next delta is an increment in the capacity var
      res = comp.get_capacity_var()
      sign = -1.0 if res in comp.get_inputs() else 1.0
      incr = increments[res]
      recpr(1, 'Testing if incrementing component {} by ({} = {}) can add value ...'.format(clr(comp.name, 'COMP'), clr(res, 'RES'), incr))
      recpr(1, 'Adding scenario {} ...'.format(clr(new_scenario.name, 'SCN')))
      try:
        new_scenario = _dispatch_component_request({res: sign * incr}, comp, new_scenario, recursion=1)
      except ComponentAtCapacityError:
        scenario.remove_branch(new_scenario)
        continue
      # now, with newly-updated production, rebalance this time step.
      try:
        new_scenario = _balance_resources_timestep(new_scenario, recursion=1)
      except NoScenarioError:
        recpr(1, 'Scenario {} was not resolvable for creating additional profit.'.format(clr(new_scenario.name, 'SCN')))
        scenario.remove_branch(new_scenario)
        continue
      # determine cost of scenario
      profit = _get_scenario_cost(new_scenario, 1)
      if profit <= nominal_profit:
        recpr(1, 'Scenario {} was not profitable compared to its parent scenario.'.format(clr(new_scenario.name, 'SCN')))
        scenario.remove_branch(new_scenario)
        continue
      if best_option['scenario'] is None or profit > best_option['profit']:
        recpr(1, 'Scenario {} is more profitable than its parent scenario! Increase: {}, Old: {}, New: {}'.format(clr(new_scenario.name, 'SCN'), profit - nominal_profit, nominal_profit, profit))
        best_option['scenario'] = new_scenario
        best_option['profit'] = profit
      # continue collecting scenarios until we have them all
    # now that scenarios are collected, keep the most profitable!
    if best_option['scenario'] is None:
      # there were no better choices than the nominal, so no changes needed.
      recpr(0, 'No additions were found to provide more value than the current scenario.')
      finished = True
    else:
      scenario.collapse_branches(best_option['scenario'].name)
  print('-'*100)
  print('Step {}: {} has been dispatched to maximize profitability.'.format(clr('???', 'TIME'), clr('???', 'TIME')))
  print('-'*100)
  return scenario

###################
# RECURSIVE LOOPS #
###################
def _balance_resources_timestep(scenario, recursion=0):
  """ balance the imbalances in the scenario, using quantized steps in each resource """
  # DEBUGG
  satisfied = False
  while not satisfied: # functionally loops over resources to balance
    # existing balance for this scenario, given previous/recursive assumptions
    #print_scenario(recursion, scenario)
    #if recursion == 0:
    # recpr(recursion, 'Reached , uh, a level, top level again, SUMMARY:')
    # print_scenario(recursion, scenario)
    # which resources are not yet balanced for this scenario?
    unbalanced = scenario.get_unbalanced('delta')
    if check_verb(recursion, 'dispatch'):
      balance = scenario.get_balance('delta')
      recpr(recursion, 'Remaining imbalanced resources: for {}'.format(clr(scenario.name, 'SCN')))
      recdfpr(recursion, balance.to_string())
    ## if balanced, then return the scenario -> it should have the updated cost on it
    if len(unbalanced) == 0:
      satisfied = True
      # this particular branch is balanced.
      recpr(recursion, 'scenario branch {} ended!'.format(clr(scenario.name, 'SCN')))
      recpr(recursion, '<--')
      #print_scenario(recursion, scenario)
      break
    # which imbalance should we satisfy?
    # nah, instead, let's assume it doesn't matter the order, and do each resource one at a time.
    # but, in case one resource isn't solvable, let's move to the next one and try it
    ## -> actually, we need to consider the cost of each one going first, because
    ##    we can't accurately determine the cost of taking contributions from the
    ##    imbalances of a parent.
    ##    -> this probably introduces some redundancy and some design NEEDS to be done.
    #next_resource = unbalanced[0]
    # options = []
    # best_option = {'cost_per': None, 'scenario': None, 'cost': None}
    # TODO how do you compare these fairly? If I have 1e6 resouce A, and 0.01 resource B,
    ##  ofc the cost of resolving A will likely be higher than B! So how do you choose which
    ##  resource to handle first?
    for next_resource in sorted(unbalanced, reverse=True):
      try:
        _satisfy_imbalance(next_resource, scenario, recursion)
        break # exit for loop, since we satisfied an imbalance
      except NoScenarioError:
        recpr(recursion, 'Resource {} was not resolvable for scenario {}'.format(clr(next_resource, 'RES'), clr(scenario.name, 'SCN')))
        # remove subscenarios
        scenario.clear_branches()
        # continue with the next unbalanced resource
        continue
    # if no resource can be solved, raise the no scenario error
    else:
      recpr(recursion, 'No resource was resolvable for scenario {}'.format(clr(next_resource, 'RES'), clr(scenario.name, 'SCN')))
      raise NoScenarioError
    # loop back
  return scenario

def _satisfy_imbalance(resource, scenario, recursion):
  """ given a resource's imbalance, satisfy it with components, one increment at a time """
  # get the imbalance that needs to be satisfied
  satisfied = False
  while not satisfied: # functionally loops over a single resource's INCREMENTS until balanced
    recpr(recursion, 'Increment Loop on {}'.format(clr(resource, 'RES')))
    imbalance = float(scenario.get_balance('delta')[resource])
    # recpr(recursion, 'DEBUGG full balance:')
    # recdfpr(recursion, scenario.get_balance('delta').to_string())
    if abs(imbalance) < 1e-10: # TODO arbitrary cutoff!
      satisfied = True
      recpr(recursion, 'imbalance net satisfied for {} resource {}!'.format(clr(scenario.name, 'SCN'), clr(resource, 'RES')))
      recpr(recursion, ' ~-~-~-~-~')
      break
    else:
      recpr(recursion, 'remaining net imbalance for {} resource {}: {}'.format(clr(scenario.name, 'SCN'), clr(resource, 'RES'), imbalance))
      if np.isnan(imbalance):
        raise RuntimeError('Got a nan!')
    sign = np.sign(imbalance) # if negative, then we need to produce more; if positive, we need to consume more
    # get the nominal increments of the resource
    nominal_increment = scenario.get_meta(dispatch=None, make_copy=False)['EGRET']['increments'].get(resource, 1) # TODO this should be defaulted somewhere else maybe?
    # the step
    delta = sign * min(abs(imbalance), abs(nominal_increment)) # delta needed is opposite sign of imbalance
    # first, see if parent scenario can provide delta desired
    #print('DEBUGG resource needed:', resource)
    #print('DEBUGG current scenario:', scenario.name)
    #print('DEBUGG parent scenario:', scenario.parent.name if scenario.parent is not None else "None")
    recpr(recursion, 'Seeking to resolve {} imbalance {} by incrementing {}'.format(clr(resource, 'RES'), imbalance, delta))
    best_option = {'cost_per': None, 'scenario': None, 'cost': None}
    from_parent = False
    if scenario.parent is not None:
    #  # check the outstanding deltas in the parent scenario, and take from that first.
      parent_balance = scenario.get_parents_deltas() # .parent.get_balance('delta')
      # what the parent has available is the sum of the parent's balance less the amount already perturbed
      parent_avail = parent_balance[resource] - scenario.get_perturbed(resource)
      # expect parent to contribute meaningfully if it's going to.
      if abs(parent_avail/delta) > 1e-5 and np.sign(parent_avail) == -sign:
        contrib = -1.0 * sign * min(abs(delta), abs(parent_avail))
        recpr(recursion, 'Parent scenario {} imbalance can contribute to resource {}: {}'.format(clr(scenario.name, 'SCN'), clr(resource, 'RES'), contrib))
        # NO! we cannot continue! We need to compare the cost of this choice
        # with getting the resource from other components!!
        #### OLD ####
        scenario.dispatch({resource: contrib}, 'parent', scenario.get_start_step()) # TODO use a better "t"! This should be passed down?
        continue
        # XXX FIXME TODO WORKING
        # create a new scenario perturbing the parent imbalance
        # count the cost of that scenario
        ### -> herein lies the problem; it's not practical to consider the cost of
        ###    the contribution from the parent scenario, as it can't be compared
        ###    on equal footing with the other component bids.
        ###    For now, consider each resource "going first" at the top level,
        ###    and compete those scenarios against each other.
        # toss that suggestion into the ring to compete against the others
        #from_parent = True
        #ssssssss
      #else:
      #  recpr(recursion, 'DEBUGG Parent cannot help with balance of', parent_avail, abs(parent_avail/delta))
    #else:
    #  recpr(recursion, 'DEBUGG no parent for', scenario.name)
      # TODO ride a negative balance, and that should be okay
    branches = _find_component_options(resource, delta, scenario, recursion+1)
    if not branches: # and not from_parent:
      raise NoScenarioError
    if len(branches) > 1:
      best_option = {'cost_per': None, 'scenario': None, 'cost': None}
      for name, branch in branches.items():
        #recpr()
        cost = _get_scenario_cost(branch, recursion)
        net = branch.get_balance('delta', ignore_perturbs=True)
        delta_actual = float(net[resource])
        #if not abs(abs(delta_actual/delta) - 1.0) < 1e-10:
        #  recpr(recursion, "Scenario {} could not completely meet the delta! {:1.3e} / {:1.3e} ({:2f} %)".format(clr(branch.name, 'SCN'), delta_actual, -delta, 100*abs(delta_actual/delta)))
        #  continue
        # recpr(recursion, 'Covered {}: {:1.2e} / {:1.2e} ({:2.2f} %)'.format(clr(resource, 'RES'), delta_actual, -1.*delta, abs(delta_actual/delta)*100))
        recpr(recursion, 'Cost of branch {}: {}'.format(clr(name, 'SCN'), cost))
        #recpr(recursion, 'DEBUGG analyzed cost is of following scenario:')
        #print_scenario(recursion, branch)
        cost_per = cost / abs(delta_actual) # NOTE do NOT abs "cost", as net-positive and net-negative are not the same!
        #print('DEBUGG delta_actual:', delta_actual)
        if best_option['cost_per'] is None or best_option['cost_per'] < cost_per:
          recpr(recursion, 'New best option branch: {} at {} per unit {} per time unit'.format(clr(name, 'SCN'), cost_per, clr(resource, 'RES')))
          best_option['scenario'] = branch
          best_option['cost_per'] = cost_per
          best_option['cost'] = cost # TODO do I need this?
        else:
          recpr(recursion, 'Rejected option branch: {} at {} per unit'.format(clr(name, 'SCN'), cost_per))
      accepted = best_option['scenario'].name
      # cost = best_option['cost']
      recpr(recursion, 'Selected branch {} as optimal given options.'.format(clr(accepted, 'SCN')))
        #cost_per = cost /
      #TODO # FIXME the rest of this "if" is wrong.
      # NOTE XXX the bid price for each branch should be PER UNIT of delta produced/consumed
      ### in order to be a fair comparison!
    ## otherwise, we don't need to branch, we can collapse up
    else:
      accepted = next(iter(branches.keys()))
      recpr(recursion, 'Selected branch {} since no alternatives available.'.format(clr(accepted, 'SCN')))
      # cost = _get_scenario_cost(scenario, recursion)
    scenario.collapse_branches(accepted)
    # update scenario cost so far # TODO useful?
    # scenario.set_cost(cost)
  # end while imbalanced
  # imbalance is satisfied, so return the scenario
  return scenario

def _get_scenario_cost(scenario, recursion):
  # TODO FIXME this is only getting activity from the first part of the time window!
  assert len(scenario.get_time_steps()) == 1
  tot_dispatch = scenario.get_dispatch('total')
  activity = tot_dispatch().loc[{'time': scenario.get_start_time()}]
  cost = 0
  levels = scenario.get_storage_levels('original')
  for comp in scenario.dispatchable + scenario.dependent:
    #recpr(recursion, 'START evaluating costs for comp', clr(comp.name, 'COMP'))
    sr = activity.loc[{'component': comp.name}].to_series()
    #sr = sr[sr != 0].dropna()
    comp_activity = sr #activity.loc[{'component': comp.name}].to_series()
    # if a storage, update meta with that information
    if comp.get_interaction().is_type('Storage'):
      scenario.get_meta(make_copy=False)['EGRET']['initial_storage_level'] = levels[comp.name]
    #print('DEBUGG comp activity: {}\n'.format(comp.name), comp_activity, type(comp_activity))
    comp_costs = comp.get_incremental_cost(comp_activity, scenario.raven_vars, scenario.get_meta(dispatch='total', make_copy=False), scenario.get_start_step())
    # pop the storage information JUST TO BE SURE
    if comp.get_interaction().is_type('Storage'):
      del scenario.get_meta(make_copy=False)['EGRET']['initial_storage_level']
    comp_tot = sum(comp_costs.values())
    cost += comp_tot
    recpr(recursion, 'Cost breakdown for scenario {} comp {}:'.format(clr(scenario.name, 'SCN'), clr(comp.name, 'COMP')))
    recpr(recursion, '  Activity:')
    for line in sr.__repr__().split('\n')[1:-1]:
      recpr(recursion, '    '+line)
    recpr(recursion, '  Cashflows:')
    for entry, value in comp_costs.items():
      recpr(recursion, '    {:>20.20s}: {:^ 1.2e}'.format(entry, value))
    recpr(recursion, '    {:>20.20s}: {:^ 1.2e}'.format('TOTAL', comp_tot))
  recpr(recursion, 'GRAND TOTAL cost of scenario "{}": {: 1.8e}'.format(clr(scenario.name, 'SCN'), cost))
  return cost

def _find_component_options(resource, delta, scenario, recursion):
  # select a component to meet the imbalance
  ### if the delta is positive, that means we need to consume, so find consumers (and vice versa)
  key = 'produced by' if delta < 0 else 'consumed by'
  can_flex = scenario.dispatchable + scenario.dependent
  eligible = (x for x in scenario.resource_map[resource][key] if x in can_flex)
  print('DEBUGG eligible:', eligible)
  req_amt = -1.0 * delta # need to counteract delta (bring it to zero)
  for comp in eligible:
    print('DEBUGG considering:', comp.name)
    # clear the new scenario variable name so there's no accidental duplication
    new_scenario = None
    # keep track of who's pinned. Storages are unique -> they can be both produce and consume
    if comp.get_interaction().is_type('Storage'):
      pin_id = comp.name + (' (providing)' if req_amt > 0 else ' (receiving)')
    else:
      pin_id = comp.name
    # don't consider ineligible components -> pinned because already used this increment
    if pin_id in scenario.pinned:
      recpr(recursion, 'Unable to use "{}" because it is pinned ...'.format(pin_id))
      continue
    # make a new scenario for picking this component
    recpr(recursion, '*'*10 + 'NEW SCENARIO branching to consider!')
    recpr(recursion, 'Checking if component {} can optimally fill need ...'.format(clr(comp.name, 'COMP')))
    new_id = "{}__{}_{}".format(comp.name, 'c' if req_amt < 0 else 'p', resource)
    new_scenario = scenario.add_branch(new_id, pin=[pin_id])
    recpr(recursion, 'pinning comp', comp.name)
    recpr(recursion, 'Added branch {}'.format(clr(new_scenario.name, 'SCN')))
    # perturb with delta as required deficit
    new_scenario.perturb_balance(resource, delta)
    try:
      new_scenario = _dispatch_component_request({resource: req_amt}, comp, new_scenario, recursion=recursion)
    except ComponentAtCapacityError:
      scenario.remove_branch(new_scenario)
      continue
    try:
      _balance_resources_timestep(new_scenario, recursion=recursion+1)
    except NoScenarioError:
      # this branch isn't viable, so clear it from the scenario
      scenario.remove_branch(new_scenario)
  return scenario.get_branches()

def _dispatch_component_request(request, comp, scenario, recursion=0):
  # TODO speedup "fresh start" option if "scenario" is guaranteed not to have any "additional dispatch" yet
  # what resource, and how much of it, are to be dispatched?
  resource, req_delta = next(iter(request.items()))
  # what is the current production level of this scenario?
  current_production_levels = scenario.get_comp_dispatch(comp.name, 'total')
  # what was the production level before this scenario started?
  original_production_levels = scenario.get_comp_dispatch(comp.name, 'original')
  # the requested dispatch is the additional request (req_delta) PLUS what is already scheduled (current production)
  full_amount = req_delta + current_production_levels[resource]
  full_request = {resource: full_amount}
  # if this comp is a storage, get its current level
  if comp.get_interaction().is_type('Storage'):
    level = scenario.get_storage_levels('total', request_comp=comp)
    recpr(recursion, 'Current storage level for "{}": {}'.format(clr(comp.name, 'COMP'), level))
  else:
    level = None
  # ask the comp to produce the requested amount (total)
  # TODO should be sending in time step, but we don't know what it is!
  assert len(scenario.get_time_steps()) == 1
  t = scenario.get_start_step()
  new_balance, meta = comp.produce(full_request, scenario.get_meta(dispatch='total', make_copy=False), scenario.raven_vars, scenario.get_dispatch('total'), t, level)
  # how much MORE or this resource got produced than was already for this scenario?
  delta_used = new_balance[resource] - current_production_levels[resource]
  ## if we didn't actually change significantly, this component is tapped out.
  if abs(delta_used / req_delta) < 1e-10: # TODO arbitrary cutoff
    if check_verb(recursion, 'dispatch'):
      recpr(recursion, 'Component {} has maxed out its capacity under the current conditions. Abandoning scenario.'.format(clr(comp.name, 'COMP')))
    raise ComponentAtCapacityError
  if check_verb(recursion, 'dispatch') and abs(req_delta - delta_used)/req_delta > 1e-10:
    recpr(recursion, 'Component {} did not fully meet the request ({}: {})'.format(clr(comp.name, 'COMP'), resource, req_delta))
    recpr(recursion, '    Instead, got', delta_used)
  recpr(recursion, 'Suggested {} TOTAL activity:'.format(clr(comp.name, 'COMP')))
  for res, val in new_balance.items():
    recpr(recursion, '    ', clr(res, 'RES'), val)
  # what is the new "additional dispatch" as a result of this new production, compared to the start of this scenario?
  if comp.get_interaction().is_type('Storage'):
    # storages are returning their additive production, but it will handle that internally later.
    ## TODO does it really need to be special or is this just the same as everyone else?
    balance_changes = new_balance
  else:
    # this is the additive level, so we combine them
    balance_changes = dict((res, new_balance[res] - original_production_levels[res]) for res in new_balance.keys())
  #print('DEBUGG new balance:', new_balance)
  #print('DEBUGG original balance:', original_production_levels)
  # update this scenario with the new additional dispatch
  scenario.update_meta(meta)
  scenario.dispatch(balance_changes, comp, t)
  return scenario

##################
# PRINTING UTILS #
##################
def check_verb(r, typ):
  if verbosity > r and typ in verb_kinds:
    return True
  return False

def print_scenario(r, scenario, drop=True, db=None, odt=None):
  if odt is None:
    odt = ['original', 'delta', 'total']
  if db is None:
    db = ['dispatch', 'balance', 'levels']
  if check_verb(r, 'dispatch'):
    border = '*'*80
    recpr(r, border, 'START')
    recpr(r, 'SCENARIO "{}" STATUS'.format(scenario.name))
    recpr(r, 'Start time: {}, End time: {}'.format(scenario.get_start_time(), scenario.get_end_time()))
    print('')
    if 'dispatch' in db:
      recpr(r,'='*60)
      recpr(r,'----- DISPATCHES -----')
      if 'original' in odt:
        recpr(r, 'Dispatch, orig:')
        recdfpr(r, scenario.get_dispatch('original').prettyprint(string=True, drop=drop))
        recpr(r,'-'*40)
        print('')
      if 'delta' in odt:
        recpr(r, 'Dispatch, delta:')
        recdfpr(r, scenario.get_dispatch('delta').prettyprint(string=True, drop=drop))
        recpr(r,'-'*40)
        print('')
      if 'total' in odt:
        recpr(r, 'Dispatch, total:')
        recdfpr(r, scenario.get_dispatch('total').prettyprint(string=True, drop=drop))
        recpr(r,'-'*40)
        print('')
    if 'balance' in db:
      recpr(r,'='*60)
      recpr(r,'----- BALANCES -----')
      print('')
      if 'original' in odt:
        recpr(r, 'Balance, orig:')
        recdfpr(r, scenario.get_balance('original').to_string())
        print('')
      if 'delta' in odt:
        recpr(r, 'Balance, delta:')
        recdfpr(r, scenario.get_balance('delta', ignore_perturbs=True).to_string())
        print('')
      if 'total' in odt:
        recpr(r, 'Balance, total:')
        recdfpr(r, scenario.get_balance('total').to_string())
    if 'balance' in db:
      recpr(r,'='*60)
      recpr(r,'----- STORAGES -----')
      print('')
      levels = scenario.get_storage_levels('total')
      if levels:
        recpr(r,'Storage Levels (total, after deltas)')
        for unit, level in levels.items():
          recpr(r,'  {}: {}'.format(unit, level))
      else:
        recpr(r, '- No storage data.')
      print('')
    recpr(r, border, 'END')
    print('', flush=True)

def recpr(r, *args):
  """ recursive printing format for tabbing in notes """
  if check_verb(r, 'dispatch'):
    print(' '+'. '*r, *args)

def recdfpr(r, strdf):
  """ recursive printing format for dataframes """
  if check_verb(r, 'dispatch'):
    for line in strdf.splitlines():
      recpr(r, line)

def clr(msg, typ):
  c = getattr(colors, typ)
  return '{}{}{}'.format(c, msg, colors.RESET)



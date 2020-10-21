
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Class for managing interactions with the Dispatchers.
"""

import os
import sys
import pickle as pk

import numpy as np

# set up path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _utils as hutils
import SerializationManager

raven_path = hutils.get_raven_loc()
sys.path.append(raven_path)
from utils import xmlUtils
sys.path.pop()

cashflow_path = hutils.get_cashflow_loc(raven_path=raven_path)
sys.path.append(cashflow_path)
from TEAL.src import CashFlows
from TEAL.src.main import run as CashFlow_run
sys.path.pop()

class DispatchRunner:
  """
    Manages the interface between RAVEN and running the dispatch
  """
  # TODO move naming templates to a common place for consistency!
  naming_template = {'comp capacity': '{comp}_capacity',
                     'dispatch var': 'Dispatch__{comp}__{res}',
                    }

  def __init__(self):
    """
      Constructor. Note instances of this are tied to single ExternalModel runs.
      @ In, None
      @ Out, None
    """
    self._case = None              # HERON case
    self._components = None        # HERON components list
    self._sources = None           # HERON sources (placeholders) list

  def load_heron_lib(self, path):
    """
      Loads HERON objects from library file.
      @ In, path, str, path (including filename) to HERON library
      @ Out, None
    """
    case, components, sources = SerializationManager.load_heron_lib(path, retry=6)
    # arguments
    self._case = case              # HERON case
    self._components = components  # HERON components list
    self._sources = sources        # HERON sources (placeholders) list
    # derivative
    self._dispatcher = self._case.dispatcher

  def extract_variables(self, raven, raven_dict):
    """
      Extract variables from RAVEN and apply them to HERON objects
      @ In, raven, object, RAVEN external model object
      @ In, raven_dict, dict, RAVEN input dictionary
      @ Out, pass_vars, dict, variables to pass to dispatcher
    """
    pass_vars = {}
    history_structure = {}
    # investigate sources for required ARMA information
    for source in self._sources:
      if source.is_type('ARMA'):
        # get structure of ARMA
        vars_needed = source.get_variable()
        for v in vars_needed:
          pass_vars[v] = getattr(raven, v)

    # get the key to mapping RAVEN multidimensional variables
    if hasattr(raven, '_indexMap'):
      pass_vars['_indexMap'] = raven._indexMap[0] # 0 is only because of how RAVEN EnsembleModel handles variables
      # collect all indices # TODO limit to those needed by sources?
      for target, required_indices in pass_vars['_indexMap'].items():
        for index in (i for i in required_indices if i not in pass_vars):
          pass_vars[index] = getattr(raven, index)
    else:
      # NOTE this should ONLY BE POSSIBLE if no ARMAs are in use!
      pass
      # FIXME index isn't always "time" ...
      #time = getattr(raven, 'time', None)
      #if time is not None:
      #  pass_vars['time'] = time

    # variable for "time" discretization, if present
    time_var = self._case.get_time_name()
    time_vals = getattr(raven, time_var, None)
    if time_vals is not None:
      pass_vars[time_var] = time_vals

    # TODO magic keywords (e.g. verbosity, MAX_TIMES, MAX_YEARS, ONLY_DISPATCH, etc)
    # TODO other arbitrary constants, such as sampled values from Outer needed in Inner?

    # component capacities
    for comp in self._components:
      name = self.naming_template['comp capacity'].format(comp=comp.name)
      update_capacity = raven_dict.get(name) # TODO is this ever not provided?
      comp.set_capacity(update_capacity)
      pass_vars[f'{comp.name}_capacity'] = update_capacity
    # TODO other case, component properties

    # load ARMA signals
    for source in self._sources:
      if source.is_type('ARMA'):
        vars_needed = source.get_variable()
        for v in vars_needed:
          pass_vars[v] = getattr(raven, v)
    return pass_vars

  def run(self, raven_vars):
    """
      Runs dispatcher.
      @ In, raven_vars, dict, dictionary of variables from raven to pass through
      @ Out, all_dispatch, DispatchState, results of dispatching
      @ Out, metrics, dict, economic metric results
    """
    ## TODO move as much of this as possible to "intialize" instead of "run"!
    # build meta variable
    ## this will be passed to external functions
    heron_meta = {}
    heron_meta['Case'] = self._case
    heron_meta['Components'] = self._components
    heron_meta['Sources'] = self._sources
    heron_meta['RAVEN_vars_full'] = raven_vars
    # build indexer for components
    ## indexer is as {component: {res: index}} where index is a standardized index for tracking activity
    heron_meta['resource_indexer'] = dict((comp, dict((res, r) for r, res in enumerate(comp.get_resources()))) for comp in self._components)
    # store meta
    meta = {'HERON': heron_meta}

    # do some checking a priori
    ## TODO can we do any of this at compile time instead of run time?
    ## -> probably, by investigating the ARMA file to be used
    self._check_time(raven_vars)
    self._check_signals(raven_vars)

    # determine analysis structure
    all_structure = self._get_structure(raven_vars)
    # just need the summary info for now
    structure = all_structure['summary']
    # set up evaluation loops
    if structure['clusters']:
      seg_type = 'Cluster'
      segs = structure['clusters']
    elif structure['segments']:   # FIXME TODO
      seg_type = 'Segment'
      segs = structure['Segment']
    else:
      seg_type = 'All'
      segs = range(1)

    interp_years = range(*structure['interpolated'])
    project_life = hutils.get_project_lifetime(self._case, self._components) - 1 # 1 for construction year
    # if the ARMA is a single year, no problem, we replay it for each year
    # if the ARMA is the same or greater number of years than the project_life, we can use the ARMA still
    # otherwise, there's a problem
    if 1 < len(interp_years) < project_life:
      raise IOError(f'An interpolated ARMA ROM was used, but there are less interpolated years ({list(range(*structure["interpolated"]))}) ' +
                    f'than requested project years ({project_life})!')

    all_dispatch = self._do_dispatch(meta, all_structure, project_life, interp_years, segs, seg_type)
    metrics = self._do_cashflow(meta, all_dispatch, all_structure, project_life, interp_years, segs, seg_type)
    return all_dispatch, metrics

  def _do_dispatch(self, meta, all_structure, project_life, interp_years, segs, seg_type):
    """
      perform dispatching
      @ In, meta, dict, dictionary of passthrough variables
      @ In, project_life, int, total analysis years (e.g. 30)
      @ In, interp_years, list, actual analysis tagged years (e.g. range(2015, 2045))
      @ In, segs, list, segments/clusters/divisions
      @ In, seg_type, str, "segment" or "cluster" if segmented or clustered
      @ Out, dispatch_results, list(DispatchState), results of dispatch for each segment/cluster and year
    """
    structure = all_structure['summary']
    num_segs = len(segs)
    active_index = {}
    # dispatch storage
    dispatch_results = {}
    # for each project year ...
    for year in range(project_life):
      interp_year = interp_years[year] if len(interp_years) > 1 else (interp_years[0] + year)
      dispatch_results[interp_year] = []
      print(f'DEBUGG Dispatching year {interp_year} ({year+1}/{project_life}):')
      # if the ARMA is interpolated, we need to track which year we're in. Otherwise, use just the
      #     nominal first year.
      active_index['year'] = year if len(range(*structure['interpolated'])) > 1 else 0 # FIXME MacroID not year
      # for each segment/cluster ..
      for s, seg in enumerate(segs): #num_segs:
        print(f'DEBUGG ... Dispatching {seg_type} {s+1}/{num_segs}:')
        active_index['division'] = seg
        # active_index['division_obj'] = seg
        meta['HERON']['active_index'] = active_index
        # truncate signals to appropriate Year, Cluster
        ## -> chop up raven_vars for sources to corresponding segment/cluster, year, and time
        meta['HERON']['RAVEN_vars'] = self._slice_signals(all_structure, meta['HERON'])
        dispatch = self._dispatcher.dispatch(self._case,
                                             self._components,
                                             self._sources,
                                             meta)
        dispatch_results[interp_year].append({'dispatch': dispatch, 'division': seg, 'active_index': active_index,
                                     'trunc_vars': meta['HERON'].pop('RAVEN_vars')})
        print(f'DEBUGG ... ... {seg_type} {s+1}/{num_segs} dispatched!')
    return dispatch_results

  def _do_cashflow(self, meta, all_dispatch, all_structure, project_life, interp_years, segs, seg_type):
    """
      run cashflow analysis
      @ In, meta, dict, passthrough variables
      @ In, all_dispatch, list, DispatchState for each cluster/segment and year
      @ In, all_structure, dict, structure of ARMA evaluates (cluster, multiyear, etc)
      @ In, project_life, int, length of project
      @ In, interp_years, list, actual project years
      @ In, segs, list, list of segments/clusters
      @ In, seg_type, str, type of divisions (segments or clusters)
      @ Out, cf_metrics, dict, CashFlow metric evaluations for full project life
    """
    num_segs = len(segs)
    # get final econoomics objects
    # FINAL settings/components/cashflows use the multiplicity of divisions for aggregated evaluation
    final_settings, final_components = self._build_econ_objects(self._case, self._components, project_life)

    yearly_cluster_data = next(iter(all_structure['details'].values()))['clusters']
    print('DEBUGG preparing FINAL CASHFLOW ...')
    print('DEBUGG years:', project_life)
    for year in range(project_life):
      interp_year = interp_years[year] if len(interp_years) > 1 else (interp_years[0] + year)
      year_data = all_dispatch[interp_year]

      for s, seg_data in enumerate(year_data):
        seg = seg_data['division']
        print(f'DEBUGG ... CashFlow for year {interp_year} ({year+1}/{project_life}), {seg_type} {s+1}/{num_segs}:')
        # get the dispatch specific to this year, this segment
        dispatch = seg_data['dispatch']
        # get cluster info from the first source -> assumes all clustering is aligned!
        # -> find the info for this cluster -> FIXME this should be restructured so searching isn't necessary!
        clusters_info = yearly_cluster_data[interp_year] if interp_year in yearly_cluster_data else yearly_cluster_data[interp_years[0]]
        for cl_info in clusters_info: #next(iter(all_structure['details'].values()))['clusters'][interp_year]:
          if cl_info['id'] == seg:
            break
        else:
          raise RuntimeError
        # how many clusters does this one represent?
        multiplicity = len(cl_info['represents'])
        print(f'DEBUGG ... ... segment multiplicity: {multiplicity} ...')
        # build evaluation cash flows
        # LOCAL component cashflows are SPECIFIC TO A DIVISION
        _, local_comps = self._build_econ_objects(self._case, self._components, project_life)
        print('DEBUGG ... ... intradivision cashflow economics objects built ...')
        # TODO do I need something here? cluster_results, start, end = self._build_econXXX_objects(self._case, self._components) # XXX implement
        # set up the active space we want to evaluate
        meta['HERON']['active_index'] = {'year': year if len(interp_years) > 1 else 0,
                                         'division': seg,
                                        }
        # truncate RAVEN variables to the active space
        meta['HERON']['RAVEN_vars'] = self._slice_signals(all_structure, meta['HERON'])
        pivot_var = meta['HERON']['Case'].get_time_name() # TODO: Better way to get pivotParameterID?
        times = meta['HERON']['RAVEN_vars'][pivot_var]
        specific_meta = dict(meta) # TODO more deepcopy needed?
        resource_indexer = meta['HERON']['resource_indexer']
        print('DEBUGG ... ... intradivision run data isolated ...')
        print('DEBUGG ... ... beginning component cashflow loop ...')
        for comp in self._components:
          print(f'DEBUGG ... ... ... comp: {comp.name} ...')
          # get corresponding current and final CashFlow.Component
          cf_comp = local_comps[comp.name]
          final_comp = final_components[comp.name]
          # sanity check
          if comp.name != cf_comp.name: raise RuntimeError
          specific_meta['HERON']['component'] = comp
          specific_meta['HERON']['activity'] = dispatch
          specific_activity = {}
          final_cashflows = final_comp.getCashflows()
          for f, heron_cf in enumerate(comp.get_cashflows()):
            print(f'DEBUGG ... ... ... ... cashflow {f}: {heron_cf.name} ...')
            # get the corresponding CashFlow.CashFlow
            cf_cf = cf_comp.getCashflows()[f]
            final_cf = final_cashflows[f]
            # sanity continued
            if not (cf_cf.name == final_cf.name == heron_cf.name): raise RuntimeError
            # FIXME time then cashflow, or cashflow then time?
            # -> "activity" is the same for every cashflow at a point in time, but
            #    many cashflows only need to be evaluated once and we can vectorize ...
            # FIXME maybe "if activity is None" approach, so it gets filled on the first cashflow
            #    when looping through time.
            # TODO we assume Capex and Recurring Year do not depend on the Activity
            if cf_cf.type == 'Capex':
              # Capex cfs should only be constructed in the first year of the project life
              # FIXME is this doing capex once per segment, or once per life?
              if year == 0:
                params = heron_cf.calculate_params(specific_meta) # a, D, Dp, x, cost
                cf_params = {'name': cf_cf.name,
                              'mult_target': heron_cf._mult_target,
                              'depreciate': heron_cf._depreciate,
                              'alpha': params['alpha'],
                              'driver': params['driver'],
                              'reference': params['ref_driver'],
                              'X': params['scaling']
                            }
                cf_cf.setParams(cf_params)
                # because alpha, driver, etc are only set once for Capex cash flows, we can just
                # hot swap this cashflow into the final_comp, I think ...
                # I believe we can do this because Capex are division-independent? Can we just do
                #   it once instead of once per division?
                final_comp._cashFlows[f] = cf_cf
                # depreciators
                # FIXME do we need to know alpha, drivers first??
                if heron_cf._depreciate:
                  cf_cf.setAmortization('MACRS', heron_cf._depreciate)
                  deprs = cf_comp._createDepreciation(cf_cf)
                  final_comp._cashFlows.extend(deprs)
                print(f'DEBUGG ... ... ... ... ... yearly contribution: {params["cost"]: 1.9e} ...')
              else: # nothing to do if Capex and year != 0
                print(f'DEBUGG ... ... ... ... ... yearly contribution: 0 ...')
            elif cf_cf.type == 'Recurring':
              # yearly recurring only need setting up once per year
              if heron_cf.get_period() == 'year':
                params = heron_cf.calculate_params(specific_meta) # a, D, Dp, x, cost
                contrib = params['cost'] # cf_cf._yearlyCashflow
                print(f'DEBUGG ... ... ... ... ... yearly contribution: {contrib: 1.9e} ...')
                final_cf._yearlyCashflow[year + 1] += contrib # FIXME multiplicity? -> should not apply to Recurring.Yearly
              # hourly recurring need iteration over time
              elif heron_cf.get_period() == 'hour':
                for t, time in enumerate(times):
                  # fill in the specific activity for this time stamp
                  for resource, r in resource_indexer[comp].items():
                    specific_activity[resource] = dispatch.get_activity(comp, resource, time)
                  specific_meta['HERON']['time_index'] = t
                  specific_meta['HERON']['time_value'] = time
                  specific_meta['HERON']['activity'] = specific_activity # TODO does the rest need to be available?
                  # contribute to cashflow (using sum as discrete integral)
                  # NOTE that intrayear depreciation is NOT being considered here
                  params = heron_cf.calculate_params(specific_meta) # a, D, Dp, x, cost
                  contrib = params['cost'] * multiplicity
                  print(f'DEBUGG ... ... ... ... ... time {t:4d} ({time:1.9e}) contribution: {contrib: 1.9e} ...')
                  final_cf._yearlyCashflow[year+1] += contrib
              else:
                raise NotImplementedError(f'Unrecognized Recurring period for "{comp.name}" cashflow "{heron_cf.name}": {heron_cf.get_period()}')
            else:
                raise NotImplementedError(f'Unrecognized CashFlow type for "{comp.name}" cashflow "{heron_cf.name}": {cf_cf.type}')
            # end CashFlow type if
          # end CashFlow per Component loop
        # end Component loop
      # end Division/segment/cluster loop
    # end Year loop

    # CashFlow, take it away.
    print('****************************************')
    print('* Starting final cashflow calculations *')
    print('****************************************')
    raven_vars = meta['HERON']['RAVEN_vars_full']
    # DEBUGG
    print('DEBUGG CASHFLOWS')
    for comp_name, comp in final_components.items():
      print(f' ... comp {comp_name} ...')
      for cf in comp.getCashflows():
        print(f' ... ... cf {cf.name} ...')
        print(f' ... ... ... D', cf._driver)
        print(f' ... ... ... a', cf._alpha)
        print(f' ... ... ... Dp', cf._reference)
        print(f' ... ... ... x', cf._scale)
        if hasattr(cf, '_yearlyCashflow'):
          print(f' ... ... ... hourly', cf._yearlyCashflow)

    cf_metrics = CashFlow_run(final_settings, list(final_components.values()), raven_vars)

    print('****************************************')
    print('DEBUGG final cashflow metrics:')
    for k, v in cf_metrics.items():
      print('  ', k, v)
    print('****************************************')
    return cf_metrics

  def _build_econ_objects(self, heron_case, heron_components, project_life):
    """
      Generates CashFlow.CashFlow instances from HERON CashFlow instances
      Note the only reason there's a difference is because HERON needs to retain some level of
      flexibility in the parameter values until this method is called, whereas CashFlow expects
      them to be evaluated.
      @ In, heron_case, HERON Case instance, global HERON settings for this analysis
      @ In, heron_components, list, HERON component instances
      @ In, project_life, int, number of years to evaluate project
      @ Out, global_settings, CashFlow.GlobalSettings instance, settings for CashFlow analysis
      @ Out, cf_components, dict, CashFlow component instances
    """
    heron_econs = list(comp.get_economics() for comp in heron_components)
    # build global econ settings for CashFlow
    global_params = heron_case.get_econ(heron_econs)
    global_settings = CashFlows.GlobalSettings()
    global_settings.setParams(global_params)
    global_settings._verbosity = 0 # FIXME direct access, also make user option?
    # build CashFlow component instances
    cf_components = {}
    for c, cfg in enumerate(heron_econs):
      # cfg is the cashflowgroup connected to the heron component
      # get the associated heron component
      heron_comp = heron_components[c]
      comp_name = heron_comp.name
      # build CashFlow equivalent component
      cf_comp = CashFlows.Component()
      cf_comp_params = {'name': comp_name,
                        'Life_time': cfg.get_lifetime(),
                        # TODO StartTime, Repetitions, tax, inflation
                       }
      cf_comp.setParams(cf_comp_params)
      cf_components[comp_name] = cf_comp
      # create all the CashFlow.CashFlows (cf_cf) for the CashFlow.Component
      cf_cfs = []
      for heron_cf in cfg.get_cashflows():
        cf_name = heron_cf.name
        # the way to build it slightly changes depending on the CashFlow type
        if heron_cf._type == 'repeating': # FIXME protected access
          cf_cf = CashFlows.Recurring()
          cf_cf_params = {'name': cf_name,
                          'X': 1.0,
                          'mult_target': heron_cf._mult_target, # FIXME protected access
                          }
          cf_cf.setParams(cf_cf_params)
          cf_cf.initParams(project_life)
        elif heron_cf._type == 'one-time': # FIXME protected access
          cf_cf = CashFlows.Capex()
          cf_cf.name = cf_name
          cf_cf.initParams(cf_comp.getLifetime())
          # alpha, driver aren't known yet, so set those later
        else:
          raise NotImplementedError(f'Unknown HERON CashFlow Type: {heron_cf._type}')
        # store new object
        cf_cfs.append(cf_cf)
      cf_comp.addCashflows(cf_cfs)
    return global_settings, cf_components

  def save_variables(self, raven, dispatch, metrics):
    """
      generates RAVEN-acceptable variables
      Saves variables on "raven" object for returning
      @ In, raven, object, RAVEN object for setting values
      @ In, dispatch, DispatchState, dispatch values (FIXME currently unused)
      @ In, metrics, dict, economic metrics
    """
    # indexer = dict((comp, dict((res, r) for r, res in enumerate(comp.get_resources()))) for comp in self._components)
    # shape = [len(dispatch), len(next(iter(dispatch)))] # years, clusters
    template = self.naming_template['dispatch var']
    # initialize variables
    # for comp in self._components:
    #   for res, r in indexer[comp].items():
    #     name = template.format(c=comp.name, r=res)
    #     setattr(raven, name, np.zeros(shape)) # FIXME need time!
    for y, (year, year_data) in enumerate(dispatch.items()):
      for c, cluster_data in enumerate(year_data):
        dispatches = cluster_data['dispatch'].create_raven_vars(template)
        # set up index map, first time only
        if y == c == 0:
          # TODO custom names?
          raven.ClusterTime = np.asarray(cluster_data['dispatch']._times) # TODO assuming same across clusters!
          raven.Cluster = np.arange(len(year_data))
          raven.Years = np.asarray(dispatch.keys())
          if not getattr(raven, '_indexMap', None):
            raven._indexMap = np.atleast_1d({})
        for var_name, data in dispatches.items():
          # if first time, initialize data structure
          if y == c == 0:
            shape = (len(dispatch), len(year_data), len(data))
            setattr(raven, var_name, np.empty(shape)) # FIXME could use np.zeros, but slower?
          getattr(raven, var_name)[y, c] = data
          getattr(raven, '_indexMap')[0][var_name] = [self._case.get_year_name(), 'Cluster', 'ClusterTime']
        #for component in self._components:
          # TODO cheating using the numpy state
        #  dispatch = cluster_data['dispatch']
        #  resource_indices = cluster_data._resources[component]
    # TODO clustering, multiyear
    # TODO should this be a Runner method or separate?
    # template = self.naming_template['dispatch var']
    # for comp_name, data in dispatch.items():
    #   for resource, usage in data.items():
    #     name = template.format(comp=comp_name, res=resource)
    #     setattr(raven, name, usage)
    #     # TODO indexMap?
    for metric, value in metrics.items():
      setattr(raven, metric, np.atleast_1d(value))

  def _get_structure(self, raven_vars):
    """
      interpret the clustering information from the ROM
      @ In, raven_vars, dict, variables coming from RAVEN
      @ Out, all_structure, dict, structure (multiyear, cluster/segments, etc) specifications
    """
    all_structure = {'details': {}, 'summary': {}}
    for source in self._sources:
      # only need ARMA information, not Functions
      if not source.is_type('ARMA'):
        continue
      structure = {}
      all_structure['details'][source] = structure
      name = source.name
      obj = pk.load(open(source._target_file, 'rb'))
      meta = obj.writeXML().getRoot()

      # interpolation
      itp_node = meta.find('InterpolatedMultiyearROM') # FIXME isn't this multicycle sometimes?
      if itp_node:
        # read macro parameters
        macro_id = itp_node.find('MacroParameterID').text.strip()
        structure['macro'] = {'id': macro_id,
                              'num': int(itp_node.find('MacroSteps').text),
                              'first': int(itp_node.find('MacroFirstStep').text),
                              'last': int(itp_node.find('MacroLastStep').text),
                             }
        macro_nodes = meta.findall('MacroStepROM')
      else:
        macro_nodes = [meta]

      # clusters
      structure['clusters'] = {}
      for macro in macro_nodes:
        if itp_node:
          ma_id = int(macro.attrib[structure['macro']['id']])
        else:
          ma_id = 0
        clusters_info = []
        structure['clusters'][ma_id] = clusters_info
        cluster_nodes = macro.findall('ClusterROM')
        if cluster_nodes:
          # structure['clusters'] = []
          for cl_node in cluster_nodes:
            cl_info = {'id': int(cl_node.attrib['cluster']),
                       'represents': cl_node.find('segments_represented').text.split(','),
                       'indices': list(int(x) for x in cl_node.find('indices').text.split(','))
                      }
            # structure['clusters'].append(cl_info)
            clusters_info.append(cl_info)

      # TODO segments
      structure['segments'] = {}

    # TODO check consistency between ROMs?
    # for now, just summarize what we found -> take it from the first source
    summary_info = next(iter(all_structure['details'].values()))
    interpolated = (summary_info['macro']['first'], summary_info['macro']['last'] + 1) if 'macro' in summary_info else (0, 1)
    # further, also take cluster structure from the first year only
    first_year_clusters = next(iter(summary_info['clusters'].values())) if 'clusters' in summary_info else {}
    clusters = list(cl['id'] for cl in first_year_clusters)# len(summary_info['clusters']) if 'clusters' in summary_info else 0
    all_structure['summary'] = {'interpolated': interpolated,
                                'clusters': clusters,
                                'segments': 0, # FIXME XXX
                                'macro_info': summary_info['macro'],
                                'cluster_info': first_year_clusters,
                                }
                                # TODO need to add index/representivity references!
    return all_structure

  def _check_signals(self, raven_vars):
    """
      Checks the length of histories to assure they are consistent.
      TODO may not be necessary if we can interpolate histories!
      @ In, raven_vars, dict, dictionary of variables from RAVEN
      @ Out, None
    """
    # check the "signal" history length
    # TODO this isn't strictly required; we can interpolate sample ARMAs, right?
    signal_shapes = {}
    for source in (item for item in self._sources if item.is_type('ARMA')):
      if not source.is_type('ARMA'):
        continue
      var_names = source.get_variable()
      if var_names is None:
        var_names = []
      for name in var_names:
        if name not in signal_shapes:
          s = raven_vars[name].shape
          signal_shapes[name] = s
    if not all((shape == s) for shape in signal_shapes.values()):
      print('History shapes:', signal_shapes)
      print('Index Map:', raven_vars.get('_indexMap', None))
      raise IOError('Synthetic histories are not of consistent shape! See "History Shapes" and "Index Map" above!')

  def _check_time(self, raven_vars):
    """
      Checks that "time" is consistent between request and provided variables
      @ In, raven_vars, dict, dictionary of variables from RAVEN
      @ Out, None
    """
    time_var = self._case.get_time_name()
    time_vals = raven_vars.get(time_var, None)
    if time_vals is not None:
      req_start, req_end, req_steps = self._dispatcher.get_time_discr()
      # check start time
      if req_start < time_vals[0]:
        raise IOError(f'Requested start time ({req_start}) is less than time variable "{time_var}" ' +
                       f'first value ({time_vals[0]})!')
      # check end time
      if req_end > time_vals[-1]:
        raise IOError(f'Requested end time ({req_start}) is greater than time variable "{time_var}" ' +
                       f'last value ({time_vals[-1]})!')
      # check number of entries
      ## TODO this shouldn't be necessary; we can interpolate!
      ##      for now, though, we don't
      if np.linspace(req_start, req_end, req_steps).size != time_vals.size:
        raise IOError('Requested number of steps ({s}) does not match "{n}" history provided ({g})!'
                       .format(n=time_var,
                               s=np.linspace(req_start, req_end, req_steps).size,
                               g=time_vals.size))

  def _slice_signals(self, all_structure, data):
    """
      Slices from full signals to specific year/cluster/segment
      Target year/cluster/segment are taken from data['active_index'] TODO change this, it's weird
      No indexes from cluster/segment or year should be present after this operation
      @ In, all_structure, dict, dictionary of informations about the source ROM structures
      @ In, data, dict, dictionary of info including RAVEN variables and HERON meta information
      @ Out, truncated, dict, RAVEN_vars portion of "data" but with truncated data
    """
    truncated = {'_indexMap': {}}
    raven = data['RAVEN_vars_full']
    index_map = raven['_indexMap']
    time_var = self._case.get_time_name()

    # are we dealing with time, interpolation, clusters, segments?
    req_indices = data['active_index']
    macro = req_indices['year'] # FIXME Macro ID!
    division = req_indices['division']
    # we keep all of Time, no divider necessary

    # build a list to hold slicing information; this will eventually be converted into a slice object
    # TODO assuming all vars have the same index structure
    summary = all_structure['summary']
    slicer_len = 1 # always time
    if len(summary['clusters']) > 1:
      slicer_len += 1 # add one if clustered
    elif summary['segments'] > 1:
      slicer_len += 1 # add one if segmented
    elif summary['interpolated']:
      slicer_len += 1
    slicer = [np.s_[:]] * slicer_len # by default, take everything
    # TODO am I overwriting this slicer in a bad way?

    for entry in raven:
      if entry in index_map:
        index_order = list(index_map[entry])
        # time -> take it all, no action needed
        # time_index = index_order.index(time_var)
        # cluster
        if 'Cluster' in index_order:
          cluster_index = index_order.index('Cluster')
          slicer[cluster_index] = division
        # macro time (e.g. cycle, year)
        if self._case.get_year_name() in index_order:
          macro_index = index_order.index(self._case.get_year_name())
          slicer[macro_index] = macro
        truncated[entry] = raven[entry][slicer]
        truncated['_indexMap'][entry] = [time_var] # the only index left should be "time"
      else:
        # entry doesn't depend on year/cluster, so keep it as is
        truncated[entry] = raven[entry]
    truncated['_indexMap'] = np.atleast_1d(truncated['_indexMap'])
    return truncated


def run(raven, raven_dict):
  """
    # TODO split into dispatch manager class and dispatch runner external model
    API for external models.
    This is run as part of the INNER ensemble model, run after the synthetic history generation
    @ In, raven, object, RAVEN variables object
    @ In, raven_dict, dict, additional RAVEN information
    @ Out, None
  """
  path = os.path.join(os.getcwd(), '..', 'heron.lib') # TODO custom name?
  # build runner
  runner = DispatchRunner()
  # load library file
  runner.load_heron_lib(path)
  # load data from RAVEN
  raven_vars = runner.extract_variables(raven, raven_dict)
  # TODO clustering, multiyear, etc?
  dispatch, metrics = runner.run(raven_vars)
  runner.save_variables(raven, dispatch, metrics)
  # TODO these are extraneous, remove from template!
  raven.time_delta = 0


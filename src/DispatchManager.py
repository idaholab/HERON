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
from CashFlow.src import CashFlows
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
      @ Out, None?
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
    project_life = hutils.get_project_lifetime(self._case, self._components)
    # if the ARMA is a single year, no problem, we replay it for each year
    # if the ARMA is the same or greater number of years than the project_life, we can use the ARMA still
    # otherwise, there's a problem
    if 1 < len(interp_years) < project_life:
      raise IOError(f'An interpolated ARMA ROM was used, but there are less interpolated years ({structure["interpolated"]}) ' +
                    f'than requested project years ({project_life})!')

    all_dispatch = self._do_dispatch(meta, all_structure, project_life, interp_years, segs, seg_type)
    self._do_cashflow(meta, all_dispatch, all_structure, project_life, interp_years, segs, seg_type)
        # given
    # TODO collect data per year/cluster/etc
    tttttt
    return dispatch

  def _do_dispatch(self, meta, all_structure, project_life, interp_years, segs, seg_type):
    """ perform dispatching TODO """
    structure = all_structure['summary']
    num_segs = len(segs)
    active_index = {}
    # dispatch storage
    dispatch_results = []
    # for each project year ...
    for year in range(project_life):
      interp_year = interp_years[year] if len(interp_years) > 1 else (interp_years[0] + year)
      dispatch_results.append([])
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
        meta['HERON']['RAVEN_vars'] = self._slice_signals(all_structure, meta['HERON'])
        ## TODO create "time" variable?
        ## -> chop up raven_vars for sources to corresponding segment/cluster, year, and time
        # FIXME XXX WORKING
        dispatch = self._dispatcher.dispatch(self._case,
                                             self._components,
                                             self._sources,
                                             meta)
        dispatch_results[-1].append({'dispatch': dispatch, 'division': seg, 'active_index': active_index,
                                     'trunc_vars': meta['HERON'].pop('RAVEN_vars')})
        print(f'DEBUGG ... ... {seg_type} {s+1}/{num_segs} dispatched!')
    return dispatch_results

  def _do_cashflow(self, meta, all_dispatch, all_structure, project_life, interp_years, segs, seg_type):
    """ run cashflow analysis TODO """
    num_segs = len(segs)
    # get final economics objects
    final_settings, final_components = self._build_econ_objects(self._case, self._components, project_life)

    print('DEBUGG running FINAL CASHFLOW ...')
    for year in range(project_life):
      interp_year = interp_years[year] if len(interp_years) > 1 else (interp_years[0] + year)
      year_data = all_dispatch[year]

      for s, seg_data in enumerate(year_data):
        seg = seg_data['division']
        print(f'DEBUGG ... CashFlow for year {interp_year} ({year+1}/{project_life}), {seg_type} {s+1}/{num_segs}:')
        # get the dispatch specific to this year, this segment
        dispatch = seg_data['dispatch']
        # get cluster info from the first source -> assumes all clustering is aligned!
        # -> find the info for this cluster -> FIXME this should be restructured so searching isn't necessary!
        for cl_info in next(iter(all_structure['details'].values()))['clusters'][interp_year]:
          if cl_info['id'] == seg:
            break
        else:
          raise RuntimeError
        # how many clusters does this one represent?
        multiplicity = len(cl_info['represents'])
        # build evaluation cash flows
        _, local_comps = self._build_econ_objects(self._case, self._components, project_life)
        print('DEBUGG ... ... intradivision cashflow economics objects built ...')
        cluster_results, start, end = self._build_economics_objects(self._case, self._components) # XXX implement
        print('DEBUGG ... ... intradivision run data isolated ...')
        # set up the active space we want to evaluate
        meta['HERON']['active_index'] = {'year': year if len(interp_years) > 1 else 0,
                                         'division': seg,
                                        }
        # truncate RAVEN variables to the active space
        meta['HERON']['RAVEN_vars'] = self._slice_signals(all_structure, meta['HERON'])
        times = meta['HERON']['Time'] # TODO how do I know what pivot var?
        specific_meta = dict(meta) # TODO more deepcopy needed?
        resource_indexer = meta['HERON']['resource_indexer']
        TODO # evaluate cash flow params HERE, or do it in the loop?
        for c, comp in self._components:
          # get corresponding current and final CashFlow.Component
          cf_comp = local_comps[c]
          final_comp = final_components[c]
          # sanity check
          if comp.name != cf_comp.name: raise RuntimeError
          specific_meta['HERON']['component'] = comp
          final_cashflows = final_comp.get_cashflows()
          for f, heron_cf in enumerate(comp.get_cashflows()):
            # get the corresponding CashFlow.CashFlow
            cf_cf = cf_comp.get_cashflows()[f]
            final_cf = final_cashflows[f]
            # sanity continued
            if not (cf_cf.name == final_cf.name == heron_cf.name): raise RuntimeError
            # the way we fill the final CashFlows depends on what type they are
            # Recurring hourly: set up numerical integral of a*D
            # Recurring yearly: contribute directly to total
            # One-time: just set the CashFlow, and only do it once
            if cf_cf.type == 'Recurring':
              if heron_cf.get_period() == 'hour':
                # hourly period needs to be multiplied by the representativity of the cluster
                year_cf = cf._yearly_cashflow[year] # FIXME we haven't calculated the values yet!!!!
                final_cf._yearly_cashflow[y+1] += year_cf
            params =
            for t, time in enumerate(times):
              for resource, r in resource_indexer[comp].items():
                specific_activity[resource] = dispatch.get_activity(comp, resource, time)
              specific_meta['HERON']['time_index'] = t
              specific_meta['HERON']['time_value'] = time
              specific_meta['HERON']['activity'] = dispatch

    # XXX FIXME WORKING TODO
    # can I maybe call a version of Dispatch._compute_cashflows, then keep all the info separate
    # instead of collapsing it? Or maybe modularize a bit?

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
      @ Out, cf_components, list, CashFlow component instances
    """
    heron_econs = list(comp.get_economics() for comp in heron_components)
    # build global econ settings for CashFlow
    global_params = heron.case.get_econ(heron_econs)
    global_settings = CashFlows.GlobalSettings()
    global_settings.set_params(global_params)
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
      cf_comp.set_params(cf_comp_params)
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
          cf_cf.set_params(cf_cf_params)
          cf_cf.init_params(project_life + 1)
        elif heron_cf._type == 'one_time': # FIXME protected access
          cf_cf = CashFlows.Capex()
          cf_cf.name = cf_name
          cf_cf.init_params(cf_comp.get_lifetime())
          # alpha, driver aren't known yet, so set those later
        else:
          raise NotImplementedError(f'Unknown HERON CashFlow Type: {heron_cf._type}')
        # store new object
        cf_cfs.append(cf_cf)
      cf_comp.add_cashflows(cf_cfs)
    return global_settings, cf_components


  def save_variables(self, raven, dispatch):
    """ generates RAVEN-acceptable variables TODO """
    # TODO clustering, multiyear
    # TODO should this be a Runner method or separate?
    template = self.naming_template['dispatch var']
    for comp_name, data in dispatch.items():
      for resource, usage in data.items():
        name = template.format(comp=comp_name, res=resource)
        setattr(raven, name, usage)
        # TODO indexMap?

  def _get_structure(self, raven_vars):
    """ interpret the clustering information from the ROM TODO """
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
    interpolated = (summary_info['macro']['first'], summary_info['macro']['last']) if 'macro' in summary_info else (0, 1)
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
        if 'Year' in index_order:
          macro_index = index_order.index('Year') # FIXME Macro ID!
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
  """
  path = os.path.join(os.getcwd(), '..', 'heron.lib') # TODO custom name?
  # build runner
  runner = DispatchRunner()
  # load library file
  runner.load_heron_lib(path)
  # load data from RAVEN
  raven_vars = runner.extract_variables(raven, raven_dict)
  # TODO clustering, multiyear, etc?
  dispatch = runner.run(raven_vars)
  runner.save_variables(raven, dispatch)


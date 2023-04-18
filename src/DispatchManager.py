
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Class for managing interactions with the Dispatchers.
"""


import os
import sys
import pickle as pk
from time import time as run_clock

import numpy as np
from typing_extensions import final

from . import _utils as hutils
from . import SerializationManager

try:
  from ravenframework.PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
  import TEAL
except ModuleNotFoundError:
  raven_path = hutils.get_raven_loc()
  sys.path.append(raven_path)
  from ravenframework.PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
  sys.path.pop()

  cashflow_path = os.path.abspath(os.path.join(hutils.get_cashflow_loc(raven_path=raven_path), '..'))
  sys.path.append(cashflow_path)
  import TEAL

# make functions findable
sys.path.append(os.getcwd())

class DispatchRunner:
  """
    Manages the interface between RAVEN and running the dispatch
  """
  # TODO move naming templates to a common place for consistency!
  naming_template = {
    'comp capacity': '{comp}_capacity',
    'dispatch var': 'Dispatch__{comp}__{tracker}__{res}',
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
    self._override_time = None     # override for micro parameter
    self._save_dispatch = False    # if True then maintain and return full dispatch record

  #####################
  # API
  def override_time(self, new):
    """
      Sets the micro time-like parameter for dispatch optimization.
      @ In, new, list(int), arguments as for numpy linpace.
      @ Out, None
    """
    self._override_time = new

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
    if self._case.debug['enabled']:
      self._save_dispatch = True

  def extract_variables(self, raven, raven_dict):
    """
      Extract variables from RAVEN and apply them to HERON objects
      @ In, raven, object, RAVEN external model object
      @ In, raven_dict, dict, RAVEN input dictionary
      @ Out, pass_vars, dict, variables to pass to dispatcher
    """
    pass_vars = {}
    history_structure = {}
    # investigate sources for required ARMA/CSV information
    for source in self._sources:
      if source.is_type('ARMA') or source.is_type("CSV"):
        # get structure of ARMA/CSV
        vars_needed = source.get_variable()
        for v in vars_needed:
          pass_vars[v] = getattr(raven, v)

    # get the key to mapping RAVEN multidimensional variables
    if hasattr(raven, '_indexMap'):
      pass_vars['_indexMap'] = raven._indexMap[0] # 0 is only because of how RAVEN EnsembleModel handles variables
      # collect all indices # TODO limit to those needed by sources?
      for target, required_indices in pass_vars['_indexMap'].items():
        for index in filter(lambda idx: idx not in pass_vars, required_indices):
          pass_vars[index] = getattr(raven, index)
    else:
      # NOTE this should ONLY BE POSSIBLE if no ARMAs or CSVs are in use!
      pass

    # variable for "time" discretization, if present
    year_var = self._case.get_year_name()
    time_var = self._case.get_time_name()
    time_vals = getattr(raven, time_var, None)
    if time_vals is not None:
      pass_vars[time_var] = time_vals

    # TODO references to all ValuedParams should probably be registered somewhere
    # like maybe in the VPFactory, then we can loop through and look for info
    # that we know from Outer and fill in the blanks? Maybe?
    for magic in self._case.dispatch_vars.keys():
      val = getattr(raven, f'{magic}_dispatch', None)
      if val is not None:
        pass_vars[magic] = float(val)

    # component capacities
    for comp in self._components:
      name = self.naming_template['comp capacity'].format(comp=comp.name)
      update_capacity = raven_dict.get(name) # TODO is this ever not provided?
      if update_capacity is not None:
        comp.set_capacity(update_capacity)
        pass_vars[f'{comp.name}_capacity'] = update_capacity
    # TODO other case, component properties

    # check macro parameter
    if year_var in dir(raven):
      year_vals = getattr(raven, year_var)
      year_size = year_vals.size
      project_life = hutils.get_project_lifetime(self._case, self._components) - 1 # 1 for construction year
      if year_size != project_life:
        raise RuntimeError(f'Provided macro variable "{year_var}" is length {year_size}, ' +
                           f'but expected project life is {project_life}! ' +
                           f'"{year_var}" values: {year_vals}')

    # load ARMA signals
    for source in self._sources:
      if source.is_type('ARMA'):
        vars_needed = source.get_variable()
        for v in vars_needed:
          vals = getattr(raven, v, None)
          # checks
          if vals is None:
            raise RuntimeError(f'HERON: Expected ARMA variable "{v}" was not passed to DispatchManager!')
          pass_vars[v] = vals
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
    print("RUNNING HERON DISPATCH MANAGER")
    heron_meta = {}
    heron_meta['Case'] = self._case
    heron_meta['Components'] = self._components
    heron_meta['Sources'] = self._sources
    heron_meta['RAVEN_vars_full'] = raven_vars
    # build indexer for components
    ## indexer is as {component: {res: index}} where index is a standardized index for tracking activity
    heron_meta['resource_indexer'] = dict((comp, dict((res, r) for r, res in enumerate(comp.get_resources())))
                                          for comp in self._components)
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
    elif structure['segments']:
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
      raise IOError(f'An interpolated ARMA ROM was used, but there are less interpolated years ' +
                    f'({list(range(*structure["interpolated"]))}) ' +
                    f'than requested project years ({project_life})!')
    # do the dispatching
    all_dispatch, metrics = self._do_dispatch(meta, all_structure, project_life, interp_years, segs, seg_type)
    return all_dispatch, metrics

  def save_variables(self, raven, all_dispatch, metrics):
    """
      generates RAVEN-acceptable variables
      Saves variables on "raven" object for returning
      @ In, raven, object, RAVEN object for setting values
      @ In, all_dispatch, dict, dispatch values
      @ In, metrics, dict, economic metrics
    """
    template = self.naming_template['dispatch var']
    for y, (year, year_data) in enumerate(all_dispatch.items()):
      for c, (cluster, dispatch) in enumerate(year_data.items()):
        dispatches = dispatch.create_raven_vars(template)
        # set up index map, first time only
        if y == c == 0:
          # string names
          year_name = self._case.get_year_name()
          clst_name = '_ROM_Cluster'
          time_name = self._case.get_time_name()
          # number of entries for each dim
          n_year = len(all_dispatch)
          n_clst = len(year_data)
          n_time = len(dispatch._times) # NOTE assuming same across clusters!
          # set indices on raven
          setattr(raven, time_name, np.asarray(dispatch._times))
          setattr(raven, year_name, np.asarray(list(all_dispatch.keys())))
          setattr(raven, clst_name, np.arange(n_clst))
          if not getattr(raven, '_indexMap', None):
            raven._indexMap = np.atleast_1d({})
        for var_name, data in dispatches.items():
          # if first time, initialize data structure
          if y == c == 0:
            shape = (n_year, n_clst, n_time)
            setattr(raven, var_name, np.empty(shape)) # NOTE could use np.zeros, but slower?
          getattr(raven, var_name)[y, c] = data
          getattr(raven, '_indexMap')[0][var_name] = [year_name, clst_name, time_name]
    cfYears = None
    for metric, value in metrics.items():
      if metric not in ['outputType', 'all_data']:
        setattr(raven, metric, np.atleast_1d(value))
      elif metric == 'all_data':
        # store the cashflow years index cfYears
        ## implicitly assume the first cashflow has representative years
        # store each of the cashflows
        for comp, comp_data in value.items():
          for cf, cf_values in comp_data.items():
            if cfYears is None:
              cfYears = len(cf_values)
            if cf.endswith(('depreciation_tax_credit', 'depreciation')):
              name = cf
            else:
              name = f'{comp}_{cf}_CashFlow'
            setattr(raven, name, np.atleast_1d(cf_values))
    if cfYears is not None:
      setattr(raven, 'cfYears', np.arange(cfYears))

    # if component capacities weren't given by Outer, save them as part of Inner
    for comp in self._components:
      cap_name = self.naming_template['comp capacity'].format(comp=comp.name)
      if cap_name not in dir(raven):
        # TODO what value should actually be used?
        setattr(raven, cap_name, -42)

  #####################
  # UTILITIES
  def _do_dispatch(self, meta, all_structure, project_life, interp_years, segs, seg_type):
    """
      perform dispatching
      @ In, meta, dict, dictionary of passthrough variables
      @ In, project_life, int, total analysis years (e.g. 30)
      @ In, interp_years, list, actual analysis tagged years (e.g. range(2015, 2045))
      @ In, segs, list(int), segments/clusters/divisions
      @ In, seg_type, str, "segment" or "cluster" if segmented or clustered
      @ Out, dispatch_results, list(DispatchState), results of dispatch for each segment/cluster and year
    """
    structure = all_structure['summary']
    ## FINAL settings/components/cashflows use the multiplicity of divisions for aggregated evaluation
    final_settings, final_components = self._build_econ_objects(self._case, self._components, project_life)
    # enable additional cashflow outputs if in debug mode
    if self._case.debug['enabled']:
      final_settings.setParams({'Output': True})
    active_index = {}
    dispatch_results = {}
    yearly_cluster_data = next(iter(all_structure['details'].values()))['clusters']
    for year in range(project_life):
      interp_year = interp_years[year] if len(interp_years) > 1 else (interp_years[0] + year)
      if self._save_dispatch:
        dispatch_results[interp_year] = {}
      # If the ARMA is interpolated, we need to track which year we're in.
      # Otherwise, use just the nominal first year.
      active_index['year'] = year if len(range(*structure['interpolated'])) > 1 else 0 # FIXME MacroID not year
      for s, seg in enumerate(segs):
        multiplicity = self._update_meta_for_segment(meta, seg, interp_year, yearly_cluster_data,
                                                     interp_years, active_index, all_structure)
        # perform dispatch
        dispatch = self._dispatcher.dispatch(self._case, self._components, self._sources, meta)
        if self._save_dispatch:
          dispatch_results[interp_year][seg] = dispatch
        # build evaluation cash flows
        self._segment_cashflow(meta, s, seg, year, dispatch, multiplicity,
                               project_life, interp_years, all_structure, final_components)
    # TEAL, take it away.
    cf_metrics = self._final_cashflow(meta, final_components, final_settings)
    return dispatch_results, cf_metrics

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
      @ Out, teal_components, dict, CashFlow component instances
    """
    heron_econs = list(comp.get_economics() for comp in heron_components)
    # build global econ settings for CashFlow
    global_params = heron_case.get_econ(heron_econs)
    global_settings = TEAL.src.CashFlows.GlobalSettings()
    global_settings.setParams(global_params)
    global_settings._verbosity = 0 # FIXME direct access, also make user option?
    # build TEAL CashFlow component instances
    teal_components = {}
    for c, cfg in enumerate(heron_econs):
      # cfg is the cashflowgroup connected to the heron component
      # get the associated heron component
      heron_comp = heron_components[c]
      comp_name = heron_comp.name
      # build TEAL equivalent component
      teal_comp = TEAL.src.CashFlows.Component()
      teal_comp_params = {'name': comp_name,
                        'Life_time': cfg.get_lifetime(),
                        # TODO StartTime, Repetitions, custom tax/inflation rate
                       }
      teal_comp.setParams(teal_comp_params)
      teal_components[comp_name] = teal_comp
      # create all the TEAL.CashFlows (teal_cf) for the TEAL.Component
      teal_cfs = []
      for heron_cf in cfg.get_cashflows():
        cf_name = heron_cf.name
        # the way to build it slightly changes depending on the CashFlow type
        if heron_cf._type == 'repeating': # FIXME protected access
          teal_cf = TEAL.src.CashFlows.Recurring()
          # NOTE: the params are listed in order of how they're read in TEAL.CashFlows.CashFlow.setParams
          teal_cf_params = {'name': cf_name,
                          # driver: comes later
                          'tax': heron_cf._taxable,
                          'inflation': heron_cf._inflation,
                          'mult_target': heron_cf._mult_target,
                          # multiply: do we ever use this?
                          # alpha: comes later
                          # reference: not relevant for recurring
                          'X': 1.0,
                          # depreciate: not relevant for recurring
                          }
          teal_cf.setParams(teal_cf_params)
          teal_cf.initParams(project_life)
        elif heron_cf._type == 'one-time':
          teal_cf = TEAL.src.CashFlows.Capex()
          teal_cf.name = cf_name
          teal_cf_params = {'name': cf_name,
                            'driver': 1.0, # handled in segment_cashflow
                            'tax': heron_cf._taxable,
                            'inflation': heron_cf._inflation,
                            'mult_target': heron_cf._mult_target,
                            # multiply: do we ever use this?
                            'alpha': 1.0, # handled in segment_cashflow
                            'reference': 1.0, # actually handled in segment_cashflow
                            'X': 1.0,
                            # depreciate: handled in segment_cashflow
                           }
          teal_cf.setParams(teal_cf_params)
          teal_cf.initParams(teal_comp.getLifetime())
          # alpha, driver aren't known yet, so set those later
        else:
          raise NotImplementedError(f'Unknown HERON CashFlow Type: {heron_cf._type}')
        # store new object
        teal_cfs.append(teal_cf)
      teal_comp.addCashflows(teal_cfs)
    return global_settings, teal_components

  def _update_meta_for_segment(self, meta, seg, interp_year, yearly_cluster_data,
                               interp_years, active_index, all_structure) -> int:
    """
      Updates the "meta" auxiliary information variable to use info specific to the segment
      @ In, meta, dict, auxiliary information
      @ In, seg, int, id for current segment (or cluster)
      @ In, interp_year, int, identifier for active year
      @ In, yearly_cluster_data, TODO
      @ In, interp_years, TODO
      @ In, active_index, dict, active indices (including year, segment)
      @ In, all_structure, dict, structure of ARMA sample/realization
      @ Out, multiplicity, int, number of segments represented by this segment within the year
    """
    # get cluster info from the first source -> assumes all clustering is aligned!
    # -> find the info for this cluster -> FIXME this should be restructured so searching isn't necessary!
    if interp_year in yearly_cluster_data:
      clusters_info = yearly_cluster_data[interp_year]
    else:
      clusters_info = yearly_cluster_data[interp_years[0]]
    for cl_info in clusters_info:
      if cl_info['id'] == seg:
        break
    else:
      raise RuntimeError
    # how many clusters does this one represent?
    multiplicity = len(cl_info['represents'])

    # update meta for the current segment
    active_index['division'] = seg
    meta['HERON']['active_index'] = active_index
    # truncate signals to appropriate Year, Cluster
    ## -> chop up raven_vars for sources to corresponding segment/cluster, year, and time
    meta['HERON']['RAVEN_vars'] = self._slice_signals(all_structure, meta['HERON'])
    return multiplicity

  def _segment_cashflow(self, meta, s, seg, year, dispatch, multiplicity,
                        project_life, interp_years, all_structure, final_components) -> None:
    """
      Update TEAL CashFlow objects with new dispatch information for a segment
      @ In, TODO
      @ Out, None
    """
    # LOCAL component cashflows are SPECIFIC TO A DIVISION
    _, local_comps = self._build_econ_objects(self._case, self._components, project_life)
    meta['HERON']['active_index'] = {'year': year if len(interp_years) > 1 else 0, 'division': seg,}
    meta['HERON']['RAVEN_vars'] = self._slice_signals(all_structure, meta['HERON'])
    pivot_var = meta['HERON']['Case'].get_time_name()
    times = meta['HERON']['RAVEN_vars'][pivot_var]
    specific_meta = dict(meta) # TODO more deepcopy needed?
    resource_indexer = meta['HERON']['resource_indexer']
    for comp in self._components:
      # get corresponding current and final CashFlow.Component
      teal_comp = local_comps[comp.name]
      final_comp = final_components[comp.name]
      # sanity check
      if comp.name != teal_comp.name: raise RuntimeError
      specific_meta['HERON']['component'] = comp
      specific_meta['HERON']['all_activity'] = dispatch
      specific_activity = {}
      final_cashflows = final_comp.getCashflows()
      for f, heron_cf in enumerate(comp.get_cashflows()):
        # get the corresponding TEAL.CashFlow
        teal_cf = teal_comp.getCashflows()[f]
        final_cf = final_cashflows[f]
        # sanity continued
        if not (teal_cf.name == final_cf.name == heron_cf.name):
            raise RuntimeError

        ## FIXME time then cashflow, or cashflow then time?
        ## "activity" is the same for every cashflow at a point in
        ## time, but many cashflows only need to be evaluated once
        ## and we can vectorize ...

        ## FIXME maybe "if activity is None" approach, so it gets
        ## filled on the first cashflow when looping through time.

        ## TODO we assume Capex and Recurring Year do not depend
        ## on the Activity

        if teal_cf.type == 'Capex':
          # Capex cfs should only be constructed in the first  of the project life
          # FIXME is this doing capex once per segment, or once per life?
          if year == 0 and s == 0:
            params = heron_cf.calculate_params(specific_meta) # a, D, Dp, x, cost
            # NOTE: listing params in order of TEAL.CashFlows.CashFlow.setParams
            cf_params = {'name': teal_cf.name,
                         'driver': params['driver'],
                         'tax': heron_cf._taxable,
                         'inflation': heron_cf._inflation,
                         'mult_target': heron_cf._mult_target,
                         # TODO "multiply" needed? Can't think of an application right now.
                         'alpha': params['alpha'],
                         'reference': params['ref_driver'],
                         'X': params['scaling'],
                         'depreciate': heron_cf._depreciate,
                        }
            teal_cf.setParams(cf_params)

            ## Because alpha, driver, etc are only set once for capex cash flows,
            ## we can just hot swap this cashflow into the final_comp, I think ...

            ## I believe we can do this because Capex are division-independent?
            ## Can we just do it once instead of once per division?

            final_comp._cashFlows[f] = teal_cf
            # depreciators
            # FIXME do we need to know alpha, drivers first??
            if heron_cf._depreciate and teal_cf.getAmortization() is None:
              teal_cf.setAmortization('MACRS', heron_cf._depreciate)
              deprs = teal_comp._createDepreciation(teal_cf)
              final_comp._cashFlows.extend(deprs)
        elif teal_cf.type == 'Recurring':
          # yearly recurring only need setting up once per year
          if heron_cf.get_period() == 'year':
            if s == 0:
              params = heron_cf.calculate_params(specific_meta) # a, D, Dp, x, cost
              contrib = params['cost']
              final_cf._yearlyCashflow[year + 1] += contrib
          # hourly recurring need iteration over time
          elif heron_cf.get_period() == 'hour':
            for t, time in enumerate(times):
              # fill in the specific activity for this time stamp
              for track_var in comp.get_tracking_vars():
                specific_activity[track_var] = {}
                for resource, r in resource_indexer[comp].items():
                  specific_activity[track_var][resource] = dispatch.get_activity(comp, track_var, resource, time)
              specific_meta['HERON']['time_index'] = t
              specific_meta['HERON']['time_value'] = time
              # TODO does the rest need to be available?
              specific_meta['HERON']['activity'] = specific_activity
              # contribute to cashflow (using sum as discrete integral)
              # NOTE that intrayear depreciation is NOT being considered here
              params = heron_cf.calculate_params(specific_meta) # a, D, Dp, x, cost
              contrib = params['cost'] * multiplicity
              final_cf._yearlyCashflow[year+1] += contrib
          else:
            raise NotImplementedError(
                f'Unrecognized Recurring period for "{comp.name}" cashflow "{heron_cf.name}": {heron_cf.get_period()}'
            )
        else:
            raise NotImplementedError(
                f'Unrecognized CashFlow type for "{comp.name}" cashflow "{heron_cf.name}": {teal_cf.type}'
            )
        # end CashFlow type if
      # end CashFlow per Component loop
    # end Component loop

  def _final_cashflow(self, meta, final_components, final_settings) -> dict:
    """
      Perform final cashflow calculations using TEAL.
      @ In, meta, dict, auxiliary information
      @ In, final_components, list, completed TEAL component objects
      @ In, final_settings, TEAL.Settings, completed TEAL settings object
      @ Out, cf_metrics, dict, values for calculated metrics
    """
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
    # END DEBUGG
    cf_metrics = TEAL.src.main.run(final_settings, list(final_components.values()), raven_vars)
    # DEBUGG
    print('****************************************')
    print('DEBUGG final cashflow metrics:')
    for k, v in cf_metrics.items():
      if k not in ['outputType', 'all_data']:
        print('  ', k, v)
    print('****************************************')
    # END DEBUGG
    return cf_metrics

  def _get_structure(self, raven_vars):
    """
      interpret the clustering information from the ROM
      @ In, raven_vars, dict, variables coming from RAVEN
      @ Out, all_structure, dict, structure (multiyear, cluster/segments, etc) specifications
    """
    all_structure = {'details': {}, 'summary': {}}
    found = False
    assert self._sources is not None
    for source in self._sources:
      if source.is_type("ARMA"):
        structure = hutils.get_synthhist_structure(source._target_file)
        all_structure["details"][source] = structure
        found = True
        break

    if not found:
      for source in self._sources:
        if source.is_type("CSV"):
          structure = hutils.get_csv_structure(
              source._target_file,
              self._case.get_year_name(),
              self._case.get_time_name()
          )
          all_structure['details'][source] = structure
          found = True
          break

    # It's important to note here. We do not anticipate users mixing
    # ARMA & CSV sources, we also don't account for discrepancies in
    # time-steps between CSV and ARMA. Eventually we may need to modify
    # the code to allow for mixed use and determine compatibility of
    # time-steps.
    if not found:
      raise RuntimeError('No ARMA or CSV found in sources! Temporal mapping is missing.')

    # TODO check consistency between ROMs?
    # for now, just summarize what we found -> take it from the first source
    summary_info = next(iter(all_structure['details'].values()))
    interpolated = (summary_info['macro']['first'], summary_info['macro']['last'] + 1) if 'macro' in summary_info else (0, 1)
    # further, also take cluster structure from the first year only
    first_year_clusters = next(iter(summary_info['clusters'].values())) if 'clusters' in summary_info else {}
    clusters = list(cl['id'] for cl in first_year_clusters)
    all_structure['summary'] = {'interpolated': interpolated,
                                'clusters': clusters,
                                'segments': 0, # FIXME XXX
                                'macro_info': summary_info['macro'] if 'macro' in summary_info else {},
                                'cluster_info': first_year_clusters,
                                } # TODO need to add index/representivity references!
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
    if self._override_time:
      self._dispatcher.set_time_discr(self._override_time)

    if time_vals is not None:
      req_start, req_end, req_steps = self._dispatcher.get_time_discr()
      # check start time
      if req_start < time_vals[0]:
        raise IOError(f'Requested start time ({req_start}) is less than time variable "{time_var}" ' +
                       f'first value ({time_vals[0]})!')
      # check end time
      if req_end > time_vals[-1]:
        raise IOError(f'Requested end time ({req_end}) is greater than time variable "{time_var}" ' +
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

    for entry in raven:
      if entry in index_map:
        index_order = list(index_map[entry])
        # TODO this is an awkward fix for histories that are missing some of the indexes
        slicer = [np.s_[:]] * len(index_order)
        # if len(index_order) < len(slicer):
        #   slicer = slicer[:len(index_order)]
        # time -> take it all, no action needed
        # cluster
        if '_ROM_Cluster' in index_order:
          cluster_index = index_order.index('_ROM_Cluster')
          slicer[cluster_index] = division
        # macro time (e.g. cycle, year)
        if self._case.get_year_name() in index_order:
          macro_index = index_order.index(self._case.get_year_name())
          slicer[macro_index] = macro
        truncated[entry] = raven[entry][tuple(slicer)]
        truncated['_indexMap'][entry] = [time_var] # the only index left should be "time"
      else:
        # entry doesn't depend on year/cluster, so keep it as is
        truncated[entry] = raven[entry]
    truncated['_indexMap'] = np.atleast_1d(truncated['_indexMap'])
    return truncated


class DispatchManager(ExternalModelPluginBase):
  """
    A plugin to run heron.lib
  """

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize the DispatchManager plugin.
      @ In, container, object, external 'self'
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ In, inputFiles, list, not used
      @ Out, None
    """
    pass

  def _readMoreXML(self, raven, xml):
    """
      Reads additional inputs for DispatchManager
      @ In, raven, object, variable-storing object
    """
    respec = xml.find('respecTime')
    if respec is not None:
      try:
        stats = [int(x) for x in respec.text.split(',')]
        raven._override_time = stats
        np.linspace(*stats) # test it out
      except Exception:
        raise IOError('DispatchManager xml: respec values should be arguments for np.linspace! Got', respec.text)

  def run(self, raven, raven_dict):
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
    # add settings from readMoreXML
    override_time = getattr(raven, '_override_time', None)
    if override_time is not None:
      runner.override_time(override_time) # TODO setter
    dispatch, metrics = runner.run(raven_vars)
    runner.save_variables(raven, dispatch, metrics)




# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Holds the template information for creating LCOE SWEEP OPT input files.
"""
import os
import sys
import copy
import shutil
import xml.etree.ElementTree as ET
import itertools as it

import numpy as np
import dill as pk

# load utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from HERON.src.base import Base
import HERON.src._utils as hutils
sys.path.pop()

# get raven location
RAVEN_LOC = os.path.abspath(os.path.join(hutils.get_raven_loc(), "ravenframework"))
try:
  import TEAL.src
except ModuleNotFoundError:
  CF_LOC = hutils.get_cashflow_loc(raven_path=RAVEN_LOC)
  if CF_LOC is None:
    raise RuntimeError('TEAL has not been found!\n' +
                       f'Check TEAL installation for the RAVEN at "{RAVEN_LOC}"')

  sys.path.append(os.path.join(CF_LOC, '..'))
from TEAL.src.main import getProjectLength
from TEAL.src import CashFlows

sys.path.append(os.path.join(RAVEN_LOC, '..'))
from ravenframework.utils import xmlUtils
from ravenframework.InputTemplates.TemplateBaseClass import Template as TemplateBase
sys.path.pop()

# default stats abbreviations
DEFAULT_STATS_NAMES = ['expectedValue', 'sigma', 'median']
SWEEP_DEFAULT_STATS_NAMES = ['maximum', 'minimum', 'percentile', 'samples', 'variance']

# prefixes for financial metrics only
FINANCIAL_PREFIXES = ["sharpe", "sortino", "es", "VaR", "glr"]

class Template(TemplateBase, Base):
  """
    Template for lcoe sweep opt class
    This templates the workflow split into sweeping over unit capacities
    in an OUTER run while optimizing unit dispatch in a INNER run.

    As designed, the ARMA stochastic noise happens entirely on the INNER,
    for easier parallelization.
  """

  # dynamic naming templates
  TemplateBase.addNamingTemplates({'jobname'        : '{case}_{io}',
                                   'stepname'       : '{action}_{subject}',
                                   'variable'       : '{unit}_{feature}',
                                   'dispatch'       : 'Dispatch__{component}__{tracker}__{resource}',
                                   'tot_activity'   :'{stats}_TotalActivity__{component}__{tracker}__{resource}',
                                   'data object'    : '{source}_{contents}',
                                   'distribution'   : '{unit}_{feature}_dist',
                                   'ARMA sampler'   : '{rom}_sampler',
                                   'lib file'       : 'heron.lib', # TODO use case name?
                                   'cashfname'      : '_{component}{cashname}',
                                   're_cash'        : '_rec_{period}_{driverType}{driverName}',
                                   'cluster_index'  : '_ROM_Cluster',
                                   'metric_name'    : '{stats}_{econ}',
                                  })

  # template nodes
  dist_template = xmlUtils.newNode('Uniform')
  dist_template.append(xmlUtils.newNode('lowerBound'))
  dist_template.append(xmlUtils.newNode('upperBound'))

  var_template = xmlUtils.newNode('variable')
  var_template.append(xmlUtils.newNode('distribution'))

  ############
  # API      #
  ############
  def __repr__(self):
    """
      String representation of this Handler and its VP
      @ In, None
      @ Out, repr, str, string representation
    """
    msg = f'<Template Driver>'
    return msg

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Base.__init__(self, **kwargs)
    here = os.path.dirname(os.path.abspath(sys.modules[self.__class__.__module__].__file__))
    self._template_path = here
    self._template_inner_path = None
    self._template_outer_path = None
    self._template_cash_path = None
    self._template_cash = None
    self._template_inner = None
    self._template_outer = None
    self.__case = None
    self.__components = None
    self.__sources = None
    self.__sweep_vars = []

  def loadTemplate(self, path):
    """
      Loads RAVEN template files from source.
      @ In, path, str, relative path to templates
      @ Out, None
    """
    rel_path = os.path.join(self._template_path, path)
    self._template_inner_path = os.path.join(rel_path, 'inner.xml')
    self._template_outer_path = os.path.join(rel_path, 'outer.xml')
    self._template_cash_path = os.path.join(rel_path, 'cash.xml')

    self._template_inner, _ = xmlUtils.loadToTree(self._template_inner_path, preserveComments=True)
    self._template_outer, _ = xmlUtils.loadToTree(self._template_outer_path, preserveComments=True)
    self._template_cash, _ = xmlUtils.loadToTree(self._template_cash_path, preserveComments=True)

  def createWorkflow(self, case, components, sources):
    """
      Create workflow XMLs
      @ In, case, HERON case, case instance for this sim
      @ In, components, list, HERON component instances for this sim
      @ In, sources, list, HERON source instances for this sim
      @ Out, inner, XML Element, root node for inner
      @ Out, outer, XML Element, root node for outer
      @ Out, cash, XML Element, root node for cashflow input
    """
    # store pieces
    self.__case = case
    self.__components = components
    self.__sources = sources
    # initialize case economics
    case.load_econ(components)
    # load a copy of the template
    inner = copy.deepcopy(self._template_inner)
    outer = copy.deepcopy(self._template_outer)
    cash = copy.deepcopy(self._template_cash)
    # modify the templates
    inner = self._modify_inner(inner, case, components, sources)
    outer = self._modify_outer(outer, case, components, sources)
    return inner, outer

  def writeWorkflow(self, templates, destination, run=False):
    """
      Write outer and inner RAVEN workflows.
      @ In, templates, list, modified XML roots
      @ In, destination, str, path to write workflows to
      @ In, run, bool, if True then attempt to run the workflows
      @ Out, None
    """
    # TODO use destination?
    # write templates
    inner, outer = templates
    outer_file = os.path.abspath(os.path.join(destination, 'outer.xml'))
    inner_file = os.path.abspath(os.path.join(destination, 'inner.xml'))

    self.raiseAMessage('========================')
    self.raiseAMessage('HERON: writing files ...')
    self.raiseAMessage('========================')

    msg_format = 'Wrote "{1}" to "{0}/"'
    with open(outer_file, 'w') as f:
      f.write(xmlUtils.prettify(outer))
    self.raiseAMessage(msg_format.format(*os.path.split(outer_file)))

    with open(inner_file, 'w') as f:
      f.write(xmlUtils.prettify(inner))
    self.raiseAMessage(msg_format.format(*os.path.split(inner_file)))

    # write library of info so it can be read in dispatch during inner run
    data = (self.__case, self.__components, self.__sources)
    lib_file = os.path.abspath(os.path.join(destination, self.namingTemplates['lib file']))
    with open(lib_file, 'wb') as lib:
      pk.dump(data, lib)
    self.raiseAMessage(msg_format.format(*os.path.split(lib_file)))
    # copy "write_inner.py", which has the denoising and capacity fixing algorithms
    conv_src = os.path.abspath(os.path.join(self._template_path, 'write_inner.py'))
    conv_file = os.path.abspath(os.path.join(destination, 'write_inner.py'))
    shutil.copyfile(conv_src, conv_file)
    self.raiseAMessage(msg_format.format(*os.path.split(conv_file)))
    # run, if requested
    if run:
      self.runWorkflow(destination)

  ############
  # UTILS    #
  ############
  ##### OUTER #####
  # Right now we modify outer by RAVEN structure, rather than HERON features
  def _modify_outer(self, template, case, components, sources):
    """
      Defines modifications to the outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, template, xml.etree.ElementTree.Element, modified template
    """
    template.set('verbosity', case.get_verbosity())
    self._modify_outer_mode(template, case, components, sources)
    self._modify_outer_runinfo(template, case)
    self._modify_outer_vargroups(template, case, components, sources)
    self._modify_outer_databases(template, case)
    self._modify_outer_dataobjects(template, case, components)
    self._modify_outer_files(template, case, sources)
    self._modify_outer_models(template, case, components, sources)
    self._modify_outer_outstreams(template, case, components, sources)
    self._modify_outer_samplers(template, case, components)
    self._modify_outer_optimizers(template, case)
    self._modify_outer_steps(template, case, components, sources)
    return template

  def _modify_outer_mode(self, template, case, components, sources):
    """
      Defines major (entity-level) modifications to outer.xml RAVEN input file due to case mode
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    if case.get_mode() == 'sweep' or case.debug['enabled']:
      template.remove(template.find('Optimizers'))
    elif case.get_mode() == 'opt':
      template.remove(template.find('Samplers'))

  def _modify_outer_runinfo(self, template, case):
    """
      Defines modifications to the RunInfo of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    run_info = template.find('RunInfo')
    case_name = self.namingTemplates['jobname'].format(case=case.name, io='o')
    run_info.find('JobName').text = case_name
    run_info.find('WorkingDir').text = case_name
    if case.debug['enabled']:
      seq = run_info.find('Sequence')
      seq.text = 'debug'
      self._updateCommaSeperatedList(seq, 'debug_output')
    elif case.get_mode() == 'sweep':
      run_info.find('Sequence').text = 'sweep'
    elif case.get_mode() == 'opt':
      run_info.find('Sequence').text = 'optimize, plot'
    # parallel
    if case.outerParallel:
      # set outer batchsize and InternalParallel
      batchSize = run_info.find('batchSize')
      batchSize.text = f'{case.outerParallel}'
      run_info.append(xmlUtils.newNode('internalParallel', text='True'))
    if case.useParallel:
      #XXX this doesn't handle non-mpi modes like torque or other custom ones
      mode = xmlUtils.newNode('mode', text='mpi')
      mode.append(xmlUtils.newNode('runQSUB'))
      if 'memory' in case.parallelRunInfo:
        mode.append(xmlUtils.newNode('memory', text=case.parallelRunInfo.pop('memory')))
      for sub in case.parallelRunInfo:
        run_info.append(xmlUtils.newNode(sub, text=str(case.parallelRunInfo[sub])))
      run_info.append(mode)
    if case.innerParallel:
      run_info.append(xmlUtils.newNode('NumMPI', text=case.innerParallel))

  def _modify_outer_vargroups(self, template, case, components, sources):
    """
      Defines modifications to the VariableGroups of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    var_groups = template.find('VariableGroups')
    # capacities
    caps = var_groups[0]
    var_list = []

    # Add component opt vars
    for comp in components:
      comp_cap_type = comp.get_capacity(None, raw=True).type
      if comp_cap_type  not in ['Function', 'ARMA', 'SyntheticHistory', 'StaticHistory']:
        var_list.append(f'{comp.name}_capacity')

    # Add dispatch opt vars
    for var in case.dispatch_vars.keys():
      var_list.append(f'{var}_dispatch')
    caps.text = ', '.join(var_list)

    # outer results
    group_outer_results = var_groups.find(".//Group[@name='GRO_outer_results']")
    # add required defaults
    econ_metrics = case.get_econ_metrics(nametype='output')
    # loop through all economic metrics (e.g., NPV, IRR) and apply required defaults to each
    default_stats_prefixes = self._get_stats_metrics_prefixes(case, DEFAULT_STATS_NAMES)
    default_stats = [self.namingTemplates['metric_name'].format(stats=sp, econ=em) \
                     for em in econ_metrics for sp in default_stats_prefixes]
    # total activity statistics
    default_stats_tot_act = []
    for sp in default_stats_prefixes:
      for component in components:
        for tracker in component.get_tracking_vars():
          resource_list = np.sort(list(component.get_resources()))
          for resource in resource_list:
            default_stats_tot_activity = self.namingTemplates['tot_activity'].format(stats=sp, component=component.name, tracker=tracker, resource=resource)
            default_stats_tot_act.append(default_stats_tot_activity)

    has_mult_metrics = len(econ_metrics + default_stats_tot_act) > 1

    # total activity statistics and economic stats added to the outer group
    for stat in default_stats+ default_stats_tot_act:
      self._updateCommaSeperatedList(group_outer_results, stat)
    # make sure user provided statistics beyond defaults get there
    if any(stat not in DEFAULT_STATS_NAMES for stat in case.get_result_statistics()):
      stats_list = self._build_result_statistic_names(case, components) #NOTE: this loops through metrics
      for stat_name in stats_list:
        if stat_name not in default_stats + default_stats_tot_act:
            self._updateCommaSeperatedList(group_outer_results, stat_name, organize_economics=has_mult_metrics)
    # sweep mode has default variable names
    elif case.get_mode() == 'sweep':
      # loop through all economic metrics (e.g., NPV, IRR) and apply required sweep defaults to each
      sweep_stats_prefixes = self._get_stats_metrics_prefixes(case, DEFAULT_STATS_NAMES+SWEEP_DEFAULT_STATS_NAMES)
      sweep_default = [self.namingTemplates['metric_name'].format(stats=sp, econ=em) \
                      for em in econ_metrics for sp in sweep_stats_prefixes]

      # total activity statistics
      sweep_stats_tot_act = []
      for sp in sweep_stats_prefixes:
        for component in components:
          for tracker in component.get_tracking_vars():
            resource_list = np.sort(list(component.get_resources()))
            for resource in resource_list:
              sweep_stats_tot_activity = self.namingTemplates['tot_activity'].format(stats=sp, component=component.name, tracker=tracker, resource=resource)
              sweep_stats_tot_act.append(sweep_stats_tot_activity )

      for sweep_name in sweep_default + sweep_stats_tot_act:
        if sweep_name not in default_stats + default_stats_tot_act:
          self._updateCommaSeperatedList(group_outer_results, sweep_name, organize_economics=has_mult_metrics)
    # opt mode adds optimization variable if not already there
    if (case.get_mode() == 'opt') and (case.get_optimization_settings() is not None):
      new_metric_outer_results = self._build_opt_metric_out_name(case)
      if (new_metric_outer_results != 'missing') and (new_metric_outer_results not in group_outer_results.text):
        # additional results statistics have been requested, add this metric if not already present
        self._updateCommaSeperatedList(group_outer_results, new_metric_outer_results, position=0)
    # labels group
    if case.get_labels():
      case_labels = ET.SubElement(var_groups, 'Group', attrib={'name': 'GRO_case_labels'})
      case_labels.text = ', '.join([f'{key}_label' for key in case.get_labels().keys()])
    if case.debug['enabled']:
      # expected dispatch, ARMA outputs
      # -> dispatch results
      group = var_groups.find(".//Group[@name='GRO_outer_debug_dispatch']")
      for component in components:
        name = component.name
        for tracker in component.get_tracking_vars():
          resource_list = np.sort(list(component.get_resources()))
          for resource in resource_list:
            var_name = self.namingTemplates['dispatch'].format(component=name, tracker=tracker, resource=resource)
            self._updateCommaSeperatedList(group, var_name)
      group = var_groups.find(".//Group[@name='GRO_outer_debug_cashflows']")
      cfs = self._find_cashflows(components)
      group.text = ', '.join(cfs)
      # -> synthetic histories?
      group = var_groups.find(".//Group[@name='GRO_outer_debug_synthetics']")
      for source in sources:
        if source.is_type('ARMA') or source.is_type('CSV'):
          synths = source.get_variable()
          for synth in synths:
            if not group.text or synth not in group.text.split(','):
              self._updateCommaSeperatedList(group, synth)

  def _modify_outer_databases(self, template, case):
    """
      Defines modifications to the Databases of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    if case.debug['enabled']:
      DBs = xmlUtils.newNode('Databases')
      template.append(DBs)
      attrs = {'name': 'dispatch', 'readMode': 'overwrite', 'directory': ''}
      db = xmlUtils.newNode('NetCDF', attrib=attrs)
      db.append(xmlUtils.newNode('variables', text='GRO_outer_debug_dispath,GRO_outer_debug_synthetics'))
      DBs.append(db)

  def _modify_outer_dataobjects(self, template, case, components):
    """
      Defines modifications to the DataObjects of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ Out, None
    """
    DOs = template.find('DataObjects')
    # labels pass to inner
    if case.get_labels():
      for node in DOs:
        if node.get('name') == 'grid':
          input_node = node.find('Input')
          input_node.text += ', GRO_case_labels'
    # remove opt components if not used
    if case.get_mode() == 'sweep' or case.debug['enabled']:
      self._remove_by_name(DOs, ['opt_eval', 'opt_soln'])
    elif case.get_mode() == 'opt':
      self._remove_by_name(DOs, ['grid'])
    # debug mode
    if case.debug['enabled']:
      # add debug dispatch output dataset
      debug_gro = ['GRO_outer_debug_dispatch', 'GRO_outer_debug_synthetics']
      deps = {self.__case.get_time_name(): debug_gro,
              self.namingTemplates['cluster_index']: debug_gro,
              self.__case.get_year_name(): debug_gro}
      self._create_dataobject(DOs, 'DataSet', 'dispatch',
                              inputs=['scaling'],
                              outputs=debug_gro,
                              depends=deps)
      # and similar for debug cashflows
      debug_gro = ['GRO_outer_debug_cashflows']
      deps = {'cfYears': debug_gro}
      self._create_dataobject(DOs, 'HistorySet', 'cashflows', outputs=debug_gro, depends=deps)

  def _modify_outer_files(self, template, case, sources):
    """
      Defines modifications to the Files of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    files = template.find('Files')
    multiruns = template.find('Steps').findall('MultiRun')
    for run in multiruns:
      # This relies on the fact that the template is hardcoded so that
      # the MultiRun 'name' attribute is either 'optimize' or 'sweep'.
      if case.get_mode() in run.attrib['name']:
        step = run
    # modify path to inner
    inner = files.find('Input') # NOTE assuming it's the first file in the template
    inner.text = '../inner.xml'
    # add other files needed by inner (functions, armas, etc)
    for source in sources:
      if source.is_type('Function'):
        # add it to the list of things that have to be transferred
        files = template.find('Files')
        src = xmlUtils.newNode('Input', attrib={'name': source.name}, text='../'+source._source)
        files.append(src)
        # add it to the Step inputs so it gets carried along
        inp = self._assemblerNode('Input', 'Files', '', source.name)
        step.insert(0, inp)

  def _modify_outer_models(self, template, case, components, sources):
    """
      Defines modifications to the Models of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    raven = template.find('Models').find('Code')
    # executable
    raven_exec = raven.find('executable')
    raven_exec_guess = os.path.abspath(os.path.join(RAVEN_LOC, '..', 'raven_framework'))
    if os.path.exists(raven_exec_guess):
      raven_exec.text = raven_exec_guess
    elif shutil.which("raven_framework") is not None:
      raven_exec.text = "raven_framework"
    else:
      raise RuntimeError("raven_framework not in PATH and not at "+raven_exec_guess)
    # conversion script
    conv = raven.find('conversion').find('input')
    conv.attrib['source'] = '../write_inner.py'

    # NOTE: if we find any CSVs in sources, we know the structure of our inner
    # has changed quite a bit. This is because we abandon the EnsembleModel &
    # MonteCarlo samplers in favor of a ExternalModel & CustomSampler when
    # using Static Histories instead of a Synthetic History. Also we do not
    # anticpate/allow users to mix the use of Static & Synthetic Histories.
    if any(x.is_type("CSV") for x in sources):
      # NOTE: this chunk of code does the initial footwork for switching
      # to Static Histories. The rest of the required changes are completed
      # in the _modify_inner_static_history() method.
      text = 'Samplers|CustomSampler@name:mc_arma_dispatch|constant@name:{}'
      # Remove anything having to do with 'denoises', it's no longer needed.
      raven.remove(raven.find(".//alias[@variable='denoises']"))
      denoises_parent = template.find(".//constant[@name='denoises']/..")
      denoises_parent.remove(denoises_parent.find(".//constant[@name='denoises']"))
      # Remove any GRO_final_return vars that compute Sigma or Var (e.g. var_NPV)
      final_return_vars = template.find('.//VariableGroups/Group[@name="GRO_outer_results"]')
      new_final_return_vars = [var for var in final_return_vars.text.split(", ") if "std" not in var and "var" not in var]
      final_return_vars.text = ', '.join(new_final_return_vars)
    else:
      text = 'Samplers|MonteCarlo@name:mc_arma_dispatch|constant@name:{}'

    for component in components:
      name = component.name
      attribs = {'variable': f'{name}_capacity', 'type':'input'}
      new = xmlUtils.newNode('alias', text=text.format(name + '_capacity'), attrib=attribs)
      raven.append(new)

    # Now we check for any non-component dispatch variables and assign aliases
    for name in case.dispatch_vars.keys():
      attribs = {'variable': f'{name}_dispatch', 'type':'input'}
      new = xmlUtils.newNode('alias', text=text.format(name + '_dispatch'), attrib=attribs)
      raven.append(new)

    # label aliases placed inside models
    for label in case.get_labels():
      attribs = {'variable': f'{label}_label', 'type':'input'}
      new = xmlUtils.newNode('alias', text=text.format(label + '_label'), attrib=attribs)
      raven.append(new)

    # if debug, grab the dispatch output instead of the summary
    if case.debug['enabled']:
      raven.find('outputDatabase').text = 'disp_full'

    # data handling: inner to outer data format
    if case.data_handling['inner_to_outer'] == 'csv':
      # swap the outputDatabase to outputExportOutStreams
      output_node = template.find('Models').find('Code').find('outputDatabase')
      output_node.tag = 'outputExportOutStreams'
      # no need to change name, as database and outstream have the same name

  def _modify_outer_outstreams(self, template, case, components, sources):
    """
      Defines modifications to the OutStreams of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    OSs = template.find('OutStreams')
    # remove opt if not used
    if case.get_mode() == 'sweep' or case.debug['enabled']:
      self._remove_by_name(OSs, ['opt_soln'])
    elif case.get_mode() == 'opt':
      self._remove_by_name(OSs, ['sweep'])
      # update plot 'opt_path' if necessary
      new_opt_objective = self._build_opt_metric_out_name(case)
      opt_path_plot_vars = OSs.find(".//Plot[@name='opt_path']").find('vars')
      if (new_opt_objective != 'missing') and (new_opt_objective not in opt_path_plot_vars.text):
        opt_path_plot_vars.text = opt_path_plot_vars.text.replace('mean_NPV', new_opt_objective)
    # debug mode
    if case.debug['enabled']:
      # modify normal metric output
      out = OSs.findall('Print')[0]
      out.attrib['name'] = 'dispatch_print'
      out.find('source').text = 'dispatch'
      # cashflow output
      cf_print = ET.SubElement(OSs, 'Print', attrib={'name': 'cashflows'})
      src = ET.SubElement(cf_print, 'type')
      src.text = 'csv'
      src = ET.SubElement(cf_print, 'source')
      src.text = 'cashflows'

      # handle dispatch plots for debug mode
      if case.debug['dispatch_plot']:
        # dispatch plot
        out_plot = ET.SubElement(OSs, 'Plot', attrib={'name': 'dispatchPlot', 'subType': 'HERON.DispatchPlot'})
        out_plot_source = ET.SubElement(out_plot, 'source')
        out_plot_source.text = 'dispatch'
        out_plot_macro = ET.SubElement(out_plot, 'macro_variable')
        out_plot_macro.text = case.get_year_name()
        out_plot_micro = ET.SubElement(out_plot, 'micro_variable')
        out_plot_micro.text = case.get_time_name()
        out_plot_signals = ET.SubElement(out_plot, 'signals')
        signals = set()
        for source in sources:
          new = source.get_variable()
          if new is not None:
            signals.update(set(new))
        out_plot_signals.text = ', '.join(signals)
      if case.debug['cashflow_plot']:
        # cashflow plot
        cf_plot = ET.SubElement(OSs, 'Plot', attrib={'name': 'cashflow_plot', 'subType': 'TEAL.CashFlowPlot'})
        cf_plot_source = ET.SubElement(cf_plot, 'source')
        cf_plot_source.text = 'cashflows'

  def _modify_outer_samplers(self, template, case, components):
    """
      Defines modifications to the Samplers/Optimizers of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ Out, None
    """
    dists_node = template.find('Distributions')
    if case.get_mode() == 'sweep' or case.debug['enabled']:
      samps_node = template.find('Samplers/Grid')
    else:
      samps_node = template.find('Optimizers/GradientDescent')
    if case.debug['enabled']:
      samps_node.tag = 'MonteCarlo'
      samps_node.attrib['name'] = 'mc'
      init = xmlUtils.newNode('samplerInit')
      init.append(xmlUtils.newNode('limit', text=1))
      samps_node.append(init)

    # NOTE: There is a chance we removed the denoises variable earlier.
    # If it was removed, that means we are using a StaticHistory.
    if samps_node.find('.//constant[@name="denoises"]') is not None:
      samps_node.find('.//constant[@name="denoises"]').text = str(case.get_num_samples())
    # add sweep variables to input

    ## TODO: Refactor this portion with the below portion to handle
    ## all general cases instead of only two.
    for key, value in case.get_labels().items():
      var_name = self.namingTemplates['variable'].format(unit=key, feature='label')
      samps_node.append(xmlUtils.newNode('constant', text=value, attrib={'name': var_name}))

    if case.debug['enabled']:
      sampler = 'mc'
    elif case.get_mode() == 'sweep':
      sampler = 'grid'
    else:
      sampler = 'opt'

    for key, value in case.dispatch_vars.items():
      var_name = self.namingTemplates['variable'].format(unit=key, feature='dispatch')
      vals = value.get_value(debug=case.debug['enabled'])
      if isinstance(vals, list):
        dist, xml = self._create_new_sweep_capacity(key, var_name, vals, sampler)
        dists_node.append(dist)
        if case.get_mode() == 'sweep':
          samps_node.append(xml)
        else:
          samps_node.append(xml)

    for component in components:
      interaction = component.get_interaction()
      # NOTE this algorithm does not check for everthing to be swept! Future work could expand it.
      # This is approached by the labels feature above
      ## Currently checked: Component.Interaction.Capacity
      ## --> this really needs to be made generic for all kinds of valued params!
      name = component.name
      var_name = self.namingTemplates['variable'].format(unit=name, feature='capacity')
      cap = interaction.get_capacity(None, raw=True)
      # do we already know the capacity values?
      if cap.is_parametric():
        vals = cap.get_value(debug=case.debug['enabled'])
        # is the capacity variable being swept over?
        if isinstance(vals, list):
          # make new Distribution, Sampler.Grid.variable
          dist, xml = self._create_new_sweep_capacity(name, var_name, vals, sampler)
          dists_node.append(dist)
          samps_node.append(xml)
          # NOTE assumption (input checked): only one interaction per component
        # if not being swept, then it's just a fixed value.
        else:
          samps_node.append(xmlUtils.newNode('constant', text=vals, attrib={'name': var_name}))
      else:
        # this capacity will be evaluated by ARMA/Function, and doesn't need to be added here.
        pass
    if case.outerParallel == 0 and case.useParallel:
      #XXX if we had a way to calculate this ahead of time,
      # this could be done in _modify_outer_runinfo
      #Need to update the outerParallel number
      run_info = template.find('RunInfo')
      case.outerParallel = len(self.__sweep_vars) + 1
      #XXX duplicate of code in _modify_outer_runinfo
      batchSize = run_info.find('batchSize')
      batchSize.text = f'{case.outerParallel}'
      run_info.append(xmlUtils.newNode('internalParallel', text='True'))

  def _modify_outer_optimizers(self, template, case):
    """
      Defines modifications to the Optimizers of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """

    # only modify if optimization_settings is in Case
    if (case.get_mode() == 'opt') and (case.get_optimization_settings() is not None) and (not case.debug['enabled']):  # TODO there should be a better way to handle the debug case
      optimization_settings = case.get_optimization_settings()
      # TODO will the optimizer always be GradientDescent?
      opt_node = template.find('Optimizers').find(".//GradientDescent[@name='cap_opt']")
      new_opt_objective = self._build_opt_metric_out_name(case)
      # swap out objective if necessary
      opt_node_objective = opt_node.find('objective')
      if (new_opt_objective != 'missing') and (new_opt_objective != opt_node_objective.text):
        opt_node_objective.text = new_opt_objective
      # swap out samplerInit values (only type implemented now)
      sampler_init = opt_node.find('samplerInit')
      type_node = sampler_init.find('type')
      try:
        type_node.text = optimization_settings['type']
      except KeyError:
        # type was not provided, so use the default value
        opt_metric, _ = case.get_opt_metric()
        stats_metric = optimization_settings['stats_metric']['name']
        type_node.text = case.economic_metrics_meta[opt_metric]['stats_map'][stats_metric]['optimization_default']

      # swap out convergence values (only persistence implemented now)
      convergence = opt_node.find('convergence')
      persistence_node = convergence.find('persistence')
      try:
        persistence_node.text = str(optimization_settings['persistence'])
      except KeyError:
        # persistence was not provided, so use the default value
        pass

      # update convergence criteria, adding nodes as necessary
      convergence_settings = optimization_settings.get('convergence', {})
      for k, v in convergence_settings.items():
        node = convergence.find(k)  # will return None if subnode is not found
        if node is None:
          convergence.append(ET.Element(k))
          node = convergence.find(k)
        node.text = v

  def _modify_outer_steps(self, template, case, components, sources):
    """
      Defines modifications to the Steps of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    steps = template.find('Steps')
    # clear out optimization if not used
    if case.get_mode() == 'sweep' or case.debug['enabled']:
      self._remove_by_name(steps, ['optimize', 'plot'])
    elif case.get_mode() == 'opt':
      self._remove_by_name(steps, ['sweep'])
    if case.debug['enabled']:
      # repurpose the sweep multirun
      sweep = steps.findall('MultiRun')[0]
      sweep.attrib['name'] = 'debug'
      sweep.find('Sampler').attrib['type'] = 'MonteCarlo'
      sweep.find('Sampler').text = 'mc'
      # remove the BasicStats collector and printer
      to_remove = []
      for output in sweep.findall('Output'):
        if output.text in ['grid', 'sweep']:
          to_remove.append(output)
      for node in to_remove:
        sweep.remove(node)
      # add debug dispatch collector and printer
      sweep.append(self._assemblerNode('Output', 'DataObjects', 'DataSet', 'dispatch'))
      sweep.append(self._assemblerNode('Output', 'DataObjects', 'HistorySet', 'cashflows'))
      sweep.append(self._assemblerNode('Output', 'Databases', 'NetCDF', 'dispatch'))
      # add an output step to print/plot summaries
      io_step = ET.SubElement(steps, 'IOStep', attrib={'name': 'debug_output'})
      io_input_dispatch = ET.SubElement(io_step, 'Input', attrib={'class': 'DataObjects', 'type': 'DataSet'})
      io_input_dispatch.text = 'dispatch'
      io_input_cashflow = ET.SubElement(io_step, 'Input', attrib={'class': 'DataObjects', 'type': 'HistorySet'})
      io_input_cashflow.text = 'cashflows'
      io_step.append(self._assemblerNode('Output', 'OutStreams', 'Print', 'dispatch_print'))
      io_step.append(self._assemblerNode('Output', 'OutStreams', 'Print', 'cashflows'))
      if case.debug['dispatch_plot']:
        io_output_dispatch = ET.SubElement(io_step, 'Output', attrib={'class': 'OutStreams', 'type': 'Plot'})
        io_output_dispatch.text = 'dispatchPlot'
      if case.debug['cashflow_plot']:
        io_output_cashflow = ET.SubElement(io_step, 'Output', attrib={'class': 'OutStreams', 'type': 'Plot'})
        io_output_cashflow.text = 'cashflow_plot'

  def _create_new_sweep_capacity(self, comp_name, var_name, capacities, sampler):
    """
      for OUTER, creates new distribution and variable for grid/opt sampling
      @ In, comp_name, str, name of component
      @ In, var_name, str, name of capacity variable
      @ In, capacities, list, float list of capacities to sweep/opt over
      @ In, sampler, string, which sampler to assume (grid, mc, opt)
      @ Out, dist, xml.etree.ElementTree,Element, XML for distribution
      @ Out, xml, xml.etree.ElementTree,Element, XML for sampled variable
    """
    # distribution
    if 'capacity' in var_name:
      dist_name = self.namingTemplates['distribution'].format(unit=comp_name, feature='capacity')
    else:
      dist_name = self.namingTemplates['distribution'].format(unit=comp_name, feature='dispatch')
    dist = copy.deepcopy(self.dist_template)
    dist.attrib['name'] = dist_name
    min_cap = min(capacities)
    max_cap = max(capacities)
    dist.find('lowerBound').text = str(min_cap)
    dist.find('upperBound').text = str(max_cap)
    xml = copy.deepcopy(self.var_template)
    xml.attrib['name'] = var_name
    self.__sweep_vars.append(var_name)
    xml.find('distribution').text = dist_name
    if sampler == 'grid':
      caps = ' '.join(str(x) for x in sorted(capacities))
      xml.append(xmlUtils.newNode('grid', attrib={'type':'value', 'construction':'custom'}, text=caps))
    elif sampler == 'opt':
      # initial value
      delta = max_cap - min_cap
      # start at 5% away from 0
      if max_cap > 0:
        initial = min_cap + 0.05 * delta
      else:
        initial = max_cap - 0.05 * delta
      xml.append(xmlUtils.newNode('initial', text=initial))
    return dist, xml



  ##### INNER #####
  # Right now we modify inner by HERON features, rather than RAVEN structure
  def _modify_inner(self, template, case, components, sources):
    """
      Defines modifications to the inner.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, template, xml.etree.ElementTree.Element, modified template
    """
    input_filepath = os.path.abspath((os.path.dirname(__file__)))
    input_filepath = input_filepath+'/../src/DispatchManager'
    ext_model = template.find('Models').find('ExternalModel')
    #ModuleToLoad not needed for HERON.DispatchManager plugin
    if 'ModuleToLoad' in ext_model.attrib:
      ext_model.attrib.pop('ModuleToLoad')
    ext_model.set('subType','HERON.DispatchManager')
    self._modify_inner_runinfo(template, case)
    self._modify_inner_sources(template, case, components, sources)
    # NOTE: this HAS to come before modify_inner_denoisings,
    #       because we'll be copy-pasting these for each denoising --> or wait, maybe that's for the Outer to do!
    self._modify_inner_components(template, case, components)
    self._modify_inner_caselabels(template, case)
    self._modify_inner_time_vars(template, case)
    self._modify_inner_econ_metrics(template, case, components)
    self._modify_inner_result_statistics(template, case, components)
    self._modify_inner_optimization_settings(template, case)
    self._modify_inner_data_handling(template, case)
    if case.debug['enabled']:
      self._modify_inner_debug(template, case, components, sources)
    self._modify_inner_static_history(template, case, sources)
    self._modify_inner_UQ(template, case, components)
    # TODO modify based on resources ... should only need if units produce multiple things, right?
    # TODO modify CashFlow input ... this will be a big undertaking with changes to the inner.
    ## Maybe let the user change them? but then we don't control the variable names. We probably have to do it.
    return template

  def _modify_inner_static_history(self, template, case, sources):
    """
      Modify entire Inner template if using StaticHistory.

      Using a static history changes many different aspects of the outer and
      inner templates. This function assumes that it will only find ONE csv in
      sources, otherwise it will make the modifications twice and could cause errors.

      @ In, template, ET.Element, root of XML template to modify
      @ In, case, HERON.Case, case object of current simulation
      @ In, sources, List[HERON.Placeholders], data generator sources for current simulation.
      @ Out, None
    """
    # If no CSV found, this won't run
    for source in filter(lambda x: x.is_type("CSV"), sources):
      # Add CSV file reference to <Files>
      csv_file = xmlUtils.newNode('Input', attrib={'name': source.name}, text=source._target_file)
      template.find("Files").append(csv_file)

      # Update <Sequence> to read the csv into memory first
      self._updateCommaSeperatedList(template.find(".//RunInfo/Sequence"), "read_static", position=0)

      # Create a new <IOStep> that is the instructions <Sequence> will run
      new_step = xmlUtils.newNode('IOStep', attrib={'name': 'read_static'})
      new_step.append(self._assemblerNode('Input', 'Files', '', source.name))
      new_step.append(self._assemblerNode('Output', 'DataObjects', 'DataSet', 'input'))
      template.find('Steps').append(new_step)

      # Change ExternalModel type in MultiRun Steps
      multi_run = template.find('.//Steps/MultiRun[@name="arma_sampling"]')
      multi_run.find("Sampler").attrib["type"] = "CustomSampler"
      multi_run.find('.//Model[@type="EnsembleModel"]').text = "dispatch"
      multi_run.find('.//Model[@type="EnsembleModel"]').attrib['type'] = "ExternalModel"

      # Modify <Group> node containing PP statistics. Remove all STD and VAR variables.
      gro_final_return = template.find('.//VariableGroups/Group[@name="GRO_final_return"]')
      new_return_vars = [var for var in gro_final_return.text.split(", ") if "std" not in var and "var" not in var]
      gro_final_return.text = ', '.join(new_return_vars)

      # Create a new <DataObject> that will store the csv data
      ## TODO I think this will break input if multiple CSV sources
      data_objs = template.find("DataObjects")
      new_data_set = xmlUtils.newNode("DataSet", attrib={"name": "input"})
      new_data_set.append(xmlUtils.newNode("Input", text=', '.join([case.get_time_name(), case.get_year_name()])))
      new_data_set.append(xmlUtils.newNode("Output", text=', '.join(source.get_variable())))
      for var in [case.get_year_name(), case.get_time_name()]:
        new_data_set.append(xmlUtils.newNode("Index", attrib={"var": var}, text=', '.join(source.get_variable())))
      if case.debug['enabled']:
        self._updateCommaSeperatedList(
            new_data_set.find('Input'),
            self.namingTemplates['cluster_index']
        )
        new_data_set.append(
            xmlUtils.newNode(
                "Index",
                attrib={"var": self.namingTemplates['cluster_index']},
                text=', '.join(source.get_variable())
            )
        )
      data_objs.append(new_data_set)

      # Modify <Models> by removing EnsembleModel and changing ExternalModel
      models = template.find("Models")
      dispatcher = models.find('.//ExternalModel[@name="dispatch"]')
      for var in source.get_variable():
        self._updateCommaSeperatedList(dispatcher.find('variables'), var)
      if case.debug['enabled'] and self.namingTemplates['cluster_index'] not in dispatcher.find('variables').text:
        self._updateCommaSeperatedList(
            dispatcher.find('variables'),
            self.namingTemplates['cluster_index']
        )

      self.raiseAMessage("Using Static History - replacing EnsembleModel with CustomSampler strategy")
      models.remove(models.find('.//EnsembleModel[@name="sample_and_dispatch"]'))

      # Remove PP Statistics that are no longer needed
      self.raiseAMessage('Using Static History - removing unneeded post-processor statistics "sigma" & "variance"')
      post_proc = models.find(".//PostProcessor")
      for sigma_node in it.chain(post_proc.findall(".//sigma"), post_proc.findall(".//variance")):
        post_proc.remove(sigma_node)

      # Modify <Samplers> to get rid of MonteCarlo reference in favor of CustomSampler
      samps = template.find("Samplers")
      monte_carlo = samps.find('.//MonteCarlo[@name="mc_arma_dispatch"]')
      monte_carlo.insert(0, self._assemblerNode("Source", "DataObjects", "DataSet", "input"))
      for var in it.chain([case.get_year_name(), case.get_time_name()], source.get_variable()):
        # We make the bold assumption here that our CSV contains a "Year" and "Time" variable.
        var_node = xmlUtils.newNode("variable", attrib={"name": var})
        monte_carlo.append(var_node)
      monte_carlo.remove(monte_carlo.find(".//samplerInit"))
      monte_carlo.tag = "CustomSampler"
      if case.debug['enabled']:
        monte_carlo.append(
            xmlUtils.newNode(
                'variable',
                attrib={'name': self.namingTemplates['cluster_index']}
            )
        )


  def _modify_inner_caselabels(self, template, case):
    """
      Create GRO_case_labels VariableGroup if labels have been provided.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    if case.get_labels():
      var_groups = template.find('VariableGroups')
      case_labels = ET.SubElement(var_groups, 'Group', attrib={'name': 'GRO_case_labels'})
      case_labels.text = ', '.join([f'{key}_label' for key in case.get_labels().keys()])
      ## Since <label> is optional, we don't want to hard-code it into
      ## the template files. So we will create it as needed and then
      ## modify GRO_armasamples_in_scalar to contain the group.
      for node in var_groups:
        if node.get('name') in ['GRO_armasamples_in_scalar', 'GRO_dispatch_in_scalar']:
          node.text += ', GRO_case_labels'
      # Add case labels to Sampler node as well.
      mc = template.find('Samplers').find('MonteCarlo')
      for key, value in case.get_labels().items():
        label_name = self.namingTemplates['variable'].format(unit=key, feature='label')
        case_labels = ET.SubElement(mc, 'constant', attrib={'name': label_name})
        case_labels.text = value

  def _modify_inner_time_vars(self, template, case):
    """
      Modify Index var attributes of DataObjects if case._time_varname is not 'Time.'
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    # Modify dispatch groups to contain correct 'Time' and 'Year' variable.
    for group in template.find('VariableGroups').findall('Group'):
      if group.attrib['name'] in ['GRO_dispatch', 'GRO_full_dispatch_indices']:
        self._updateCommaSeperatedList(group, case.get_time_name())
        self._updateCommaSeperatedList(group, case.get_year_name())
    # Modify Data Objects to contain correct index var.
    data_objs = template.find('DataObjects')
    for index in data_objs.findall("DataSet/Index"):
      if index.get('var') == 'Time':
        index.set('var', case.get_time_name())
      if index.get('var') == 'Year':
        index.set('var', case.get_year_name())

  def _modify_inner_runinfo(self, template, case):
    """
      Defines modifications to the RunInfo of inner.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
       @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    case_name = self.namingTemplates['jobname'].format(case=case.name, io='i')
    run_info = template.find('RunInfo')
    run_info.find('JobName').text = case_name
    run_info.find('WorkingDir').text = case_name
    if case.debug['enabled']:
      # need to "write full" as part of sequence, after arma sampling
      self._updateCommaSeperatedList(run_info.find('Sequence'), 'write_full', after='arma_sampling')
    # parallel
    if case.innerParallel:
      run_info.append(xmlUtils.newNode('internalParallel', text='True'))
      run_info.find('batchSize').text = f'{case.innerParallel}'

  def _modify_inner_sources(self, template, case, components, sources):
    """
      Defines modifications to the inner.xml RAVEN input file due to Sources/Placeholders.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    # for every ARMA SOURCE, we need to:
    #  - load the source to a Model
    #  - sample it N times (N = # denoise)
    #  - do signal preprocessing and dispatch
    #  - dump the results to file? --> only for viewing, debugging, so probably yes
    for source in sources:
      if source.is_type('ARMA'):
        # add a step to load the model
        self._iostep_load_rom(template, case, components, source)
        # add a step to print the rom meta
        self._iostep_rom_meta(template, source)
        # add the source to the arma-and-dispatch ensemble
        self._add_arma_to_ensemble(template, source)
        # NOTE assuming input to all ARMAs is "scaling" constant = 1.0, already in MC sampler
        if source.eval_mode == 'clustered':
          # add _ROM_Cluster to the variable group if it isn't there already
          var_group = template.find("VariableGroups/Group")
          if self.namingTemplates['cluster_index'] not in var_group.text:
            var_group.text += f",{self.namingTemplates['cluster_index']}"
          # make sure _ROM_Cluster is part of dispatch targetevaluation
          found = False
          for dataObj in template.find('DataObjects').findall('DataSet'):
            if dataObj.attrib['name'] == 'dispatch_eval':
              dispatch_eval = dataObj
              for idx in dataObj.findall('Index'):
                if idx.attrib['var'] == self.namingTemplates['cluster_index']:
                  found = True
                  break
              break
          else:
            raise RuntimeError
          if not found:
            dispatch_eval.append(xmlUtils.newNode('Index',
                                                  attrib={'var': self.namingTemplates['cluster_index']},
                                                  text='GRO_dispatch_in_Time'))

      elif source.is_type('CSV'):
        if case.debug['enabled']:
          # add signals to dispatch_in_Time variable group
          dit = template.find('VariableGroups').find('.//Group[@name="GRO_dispatch_in_Time"]')
          for var in source.get_variable():
            if dit.text is None or var not in dit.text:
              self._updateCommaSeperatedList(dit, var)

      elif source.is_type('Function'):
        # nothing to do ... ?
        pass

  def _modify_inner_components(self, template, case, components):
    """
      Defines modifications to the inner.xml RAVEN input file due to Components.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ Out, None
    """
    mc = template.find('Samplers').find('MonteCarlo')
    # find specific variable groups
    groups = {}
    var_groups = template.find('VariableGroups')
    for tag in ['capacities', 'init_disp', 'full_dispatch']:
      groups[tag] = var_groups.find(f".//Group[@name='GRO_{tag}']")

    # change inner input due to components requested
    for component in components:
      name = component.name
      # treat capacity
      ## we just need to make sure everything we need gets into the dispatch ensemble model.
      ## For each interaction of each component, that means making sure the Function, ARMA, or constant makes it.
      ## Constants from outer (namely sweep/opt capacities) are set in the MC Sampler from the outer
      ## The Dispatch needs info from the Outer to know which capacity to use, so we can't pass it from here.
      capacity = component.get_capacity(None, raw=True)
      interaction = component.get_interaction()
      parametric = capacity.is_parametric()

      if parametric:
        # this capacity is being [swept or optimized in outer] (list) or is constant (float)
        # -> so add a node, put either the const value or a dummy in place
        cap_name = self.namingTemplates['variable'].format(unit=name, feature='capacity')
        values = capacity.get_value(debug=case.debug['enabled'])
        if isinstance(values, list):
          cap_val = 42 # placeholder
        else:
          cap_val = values
        mc.append(xmlUtils.newNode('constant', attrib={'name': cap_name}, text=cap_val))
        # add component to applicable variable groups
        self._updateCommaSeperatedList(groups['capacities'], cap_name)
      elif capacity.type in ['StaticHistory', 'SyntheticHistory', 'Function', 'Variable']:
        # capacity is limited by a signal, so it has to be handled in the dispatch; don't include it here.
        # OR capacity is limited by a function, and we also can't handle it here, but in the dispatch.
        pass
      else:
        raise NotImplementedError(f'Capacity from "{capacity}" not implemented yet. Component: {cap_name}')

      for tracker in component.get_tracking_vars():
        resource_list = np.sort(list(component.get_resources()))
        for resource in resource_list:
          var_name = self.namingTemplates['dispatch'].format(component=name, tracker=tracker, resource=resource)
          self._updateCommaSeperatedList(groups['init_disp'], var_name)
          self._updateCommaSeperatedList(groups['full_dispatch'], var_name)

  def _modify_inner_UQ(self, template, case, components):
    """
      Defines modifications to the inner.xml RAVEN input file due to UQ requiring further sampling.
      NOTE: currently assuming only new distributions are within component cashflows...
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ Out, None
    """
    mc = template.find('Samplers').find('MonteCarlo')
    if not mc:
      pass #idk, error?
    # find specific variable groups
    var_groups = template.find('VariableGroups')
    # find distributions group
    distributions = template.find('Distributions')

    # check if UQ group exists, if not create variable group
    group_UQ = var_groups.find("Group[@name='GRO_UQ']")
    if not group_UQ:
      group_UQ = xmlUtils.newNode('Group', attrib={'name': 'GRO_UQ'})

    comp_distributions = {}
    cf_attrs = ['_driver', '_alpha', '_reference', '_scale']

    # looping through components to find UQs
    for component in components:
      comp_name = component.name
      # this is gonna be gross
      cfs = component.get_cashflows()
      for cf in cfs:
        for attr in cf_attrs:
          vp = getattr(cf,attr)
          if vp.type == 'RandomVariable':
            unit_name = f'{comp_name}_{cf.name}'
            feature_name = attr.split('_')[-1]
            dist_name = self.namingTemplates['distribution'].format(unit=unit_name, feature=feature_name)
            feat_name = self.namingTemplates['variable'].format(unit=unit_name, feature=feature_name)

            dist_node = vp._vp.get_distribution() #ugh, this is NOT the XML... will have to reconstruct.
            dist_node.attrib['name'] = dist_name
            comp_distributions[dist_name] = {}
            comp_distributions[dist_name]['xml'] = dist_node
            comp_distributions[dist_name]['var'] = feat_name

    if not comp_distributions:
      return

    # These variable groups need to have GRO_UQ
    groups = {}
    for tag in ['dispatch_in_scalar', 'armasamples_in_scalar']:
      groups[tag] = var_groups.find(f".//Group[@name='GRO_{tag}']")
    # Now we have to add GroupUQ to other groups
    self._updateCommaSeperatedList(groups['dispatch_in_scalar'], 'GRO_UQ', position=1)
    self._updateCommaSeperatedList(groups['armasamples_in_scalar'], 'GRO_UQ', position=2)

    for dist_name, dist_contents in comp_distributions.items():
      # 0. add VP name to Group UQ
      self._updateCommaSeperatedList(group_UQ, dist_contents['var'])
      # 1. add the distribution to Distribution group
      distributions.append(dist_contents['xml'])
      # 2. add VP name as variable in MC sampler
      new_var = xmlUtils.newNode('variable', attrib={'name':dist_contents['var']})
      # 3. add distribution to MC variable
      new_var.append(xmlUtils.newNode('distribution', text=dist_name))
      mc.append(new_var)

    var_groups.append(group_UQ)


  def _modify_inner_debug(self, template, case, components, sources):
    """
      Modify template to work in a debug mode.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    # RunInfo
    seq = template.find('.//RunInfo/Sequence')
    seq.text = (','.join(seq.text.split(',')[:-2])) # exclude basic stats parts
    # Steps
    for step in template.find('Steps'):
      if step.get('name') == 'arma_sampling':
        step.append(self._assemblerNode('Output', 'DataObjects', 'DataSet', 'disp_full'))
    # Variable Groups
    grp = template.find('VariableGroups').find(".//Group[@name='GRO_cashflows']")
    cfs = self._find_cashflows(components)
    grp.text = ', '.join(cfs)
    # Model
    extmod_vars = template.find('Models').find('ExternalModel').find('variables')
    self._updateCommaSeperatedList(extmod_vars, 'GRO_full_dispatch')
    self._updateCommaSeperatedList(extmod_vars, 'GRO_full_dispatch_indices')
    self._updateCommaSeperatedList(extmod_vars, 'GRO_cashflows')
    self._updateCommaSeperatedList(extmod_vars, 'cfYears')
    # DataObject
    dispatch_eval = template.find('DataObjects').find(".//DataSet[@name='dispatch_eval']")
    dispatch_eval.append(xmlUtils.newNode('Output', text='GRO_full_dispatch'))
    for idx in dispatch_eval.findall('Index'):
      self._updateCommaSeperatedList(idx, 'GRO_full_dispatch')

  def _modify_inner_optimization_settings(self, template, case):
    """
      Modifies template to include optimization settings
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    # TODO currently only modifies if optimization settings has metric and/or type, add additional settings?
    # only modify if the mode is 'opt' and <optimization_settings> has anything to modify
    if (case.get_mode() == 'opt') and (case.get_optimization_settings() is not None):
      optimization_settings = case.get_optimization_settings()
      # optimization objective name provided (or 'missing')
      new_objective = self._build_opt_metric_out_name(case)
      # add optimization objective name to VariableGroups 'GRO_final_return' if not already there
      group = template.find('VariableGroups').find(".//Group[@name='GRO_final_return']")
      if group.text is None:
        self._updateCommaSeperatedList(group, new_objective, postion=0)
      elif (new_objective != 'missing') and (new_objective not in group.text):
        self._updateCommaSeperatedList(group, new_objective, position=0)
      # add optimization objective to PostProcessor list if not already there
      pp_node = template.find('Models').find(".//PostProcessor[@name='statistics']")
      if new_objective != 'missing':
        raven_metric_name = optimization_settings['stats_metric']['name']
        prefix = self._get_stats_metrics_prefixes(case, [raven_metric_name])[0]
        if pp_node.find(raven_metric_name) is None:
          opt_metric, _ = case.get_opt_metric()
          # add subnode to PostProcessor
          if 'threshold' in optimization_settings['stats_metric']:
            if raven_metric_name in ['valueAtRisk', 'expectedShortfall']:
              threshold = str(optimization_settings['stats_metric']['threshold'])
            else:
              threshold = optimization_settings['stats_metric']['threshold']
            new_node = xmlUtils.newNode(raven_metric_name, text=opt_metric,
                                        attrib={'prefix': prefix,
                                                'threshold': threshold})
          elif 'percent' in optimization_settings['stats_metric']:
            percent = str(optimization_settings['stats_metric']['percent'])
            new_node = xmlUtils.newNode(raven_metric_name, text=opt_metric,
                                        attrib={'prefix': prefix,
                                                'percent': percent})
          else:
            new_node = xmlUtils.newNode(raven_metric_name, text=opt_metric,
                                        attrib={'prefix': prefix})
          pp_node.append(new_node)
        else:
          # check that subnode has correct values
          subnode = pp_node.find(raven_metric_name)
          # check that prefix is correct
          if prefix != subnode.attrib['prefix']:
            subnode.attrib['prefix'] = prefix
          # percentile has additional parameter to check
          if 'percent' in optimization_settings['stats_metric']:
            # see if percentile already has what we need
            if str(int(optimization_settings['stats_metric']['percent'])) not in subnode.attrib['percent']:
              # nope, need to add the percent to the existing attribute
              subnode.attrib['percent'] += ','+str(optimization_settings['stats_metric']['percent'])
          if 'threshold' in optimization_settings['stats_metric']:
            # see if the threshold is already there
            if str(optimization_settings['stats_metric']['threshold']) not in subnode.attrib['threshold']:
              # nope, need to add the threshold to existing attribute
              subnode.attrib['threshold'] += ','+str(optimization_settings['stats_metric']['threshold'])
      else:
        # new_objective is missing, use mean_metric
        if pp_node.find('expectedValue') is None:
          opt_metric, _ = case.get_opt_metric()
          pp_node.append(xmlUtils.newNode('expectedValue', text=opt_metric,
                                          attrib={'prefix': 'mean'}))
        else:
          # check that the subnode has the correct values
          subnode = pp_node.find('expectedValue')
          if 'mean' != subnode.attrib['prefix']:
            subnode.attrib['prefix'] = 'mean'
    # if no optimization settings specified, make sure mean_metric is in PostProcessor node
    elif case.get_mode() == 'opt':
      pp_node = template.find('Models').find(".//PostProcessor[@name='statistics']")
      if pp_node.find('expectedValue') is None:
        opt_metric, _ = case.get_opt_metric()
        pp_node.append(xmlUtils.newNode('expectedValue', text=opt_metric,
                                        attrib={'prefix': 'mean'}))
      else:
        # check that the subnode has the correct values
        subnode = pp_node.find('expectedValue')
        if 'mean' != subnode.attrib['prefix']:
          subnode.attrib['prefix'] = 'mean'

  def _modify_inner_econ_metrics(self, template, case, components):
    """
      Modifies template to include economic metrics
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    # get all economic metrics intended for use in TEAL and reported back
    econ_metrics = case.get_econ_metrics(nametype='output')
    tot_act_vars = []
    for component in components:
      for tracker in component.get_tracking_vars():
        resource_list = np.sort(list(component.get_resources()))
        for resource in resource_list:
          tot_act_var = "TotalActivity__" + component.name + "__" + tracker + "__"+ resource
          tot_act_vars.append(tot_act_var)
    # handle VariableGroups and data objects
    var_groups = template.find('VariableGroups')
    data_objs = template.find('DataObjects')

    # find variable groups to update with economic metrics
    dispatch_out = var_groups.find(".//Group[@name='GRO_dispatch_out']")
    arma_samp_out = var_groups.find(".//Group[@name='GRO_armasamples_out_scalar']")
    # find point set output node to update with economic metrics
    arma_metrics = data_objs.find(".//PointSet[@name='arma_metrics']")
    arma_metrics_out = arma_metrics.find("Output")
    # update fields with econ metric names
    for em in econ_metrics + tot_act_vars:
      self._updateCommaSeperatedList(dispatch_out, em)
      self._updateCommaSeperatedList(arma_samp_out, em)
      self._updateCommaSeperatedList(arma_metrics_out, em)

  def _modify_inner_result_statistics(self, template, case, components):
    """
      Modifies template to include result statistics
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    # handle VariableGroups
    var_groups = template.find('VariableGroups')
    # final return variable group (sent to outer)
    group_final_return = var_groups.find(".//Group[@name='GRO_final_return']")
    # add required defaults
    econ_metrics = case.get_econ_metrics(nametype='output')

    # loop through all economic metrics (e.g., NPV, IRR) and apply required defaults to each
    default_stats_prefixes = self._get_stats_metrics_prefixes(case, DEFAULT_STATS_NAMES)
    default_stats = [self.namingTemplates['metric_name'].format(stats=sp, econ=em) \
                     for em in econ_metrics for sp in default_stats_prefixes]

    # total activity statistics
    default_stats_tot_act = []
    for sp in default_stats_prefixes:
      for component in components:
        for tracker in component.get_tracking_vars():
          resource_list = np.sort(list(component.get_resources()))
          for resource in resource_list:
            default_stats_tot_activity = self.namingTemplates['tot_activity'].format(stats=sp, component=component.name, tracker=tracker, resource=resource)
            default_stats_tot_act.append(default_stats_tot_activity)

    has_mult_metrics = len(econ_metrics + default_stats_tot_act) > 1

    for stat in default_stats + default_stats_tot_act:
      self._updateCommaSeperatedList(group_final_return, stat)
    # make sure user provided statistics beyond defaults get there
    if any(stat not in DEFAULT_STATS_NAMES for stat in case.get_result_statistics()):
      stats_list = self._build_result_statistic_names(case, components) #NOTE: this loops through metrics
      for stat_name in stats_list:
        if stat_name not in default_stats + default_stats_tot_act:
          self._updateCommaSeperatedList(group_final_return, stat_name, organize_economics=has_mult_metrics)
    # sweep mode has default variable names
    elif case.get_mode() == 'sweep':
      # loop through all economic metrics (e.g., NPV, IRR) and apply required sweep defaults to each
      sweep_stats_prefixes = self._get_stats_metrics_prefixes(case, DEFAULT_STATS_NAMES+SWEEP_DEFAULT_STATS_NAMES)
      sweep_default = [self.namingTemplates['metric_name'].format(stats=sp, econ=em) \
                       for em in econ_metrics for sp in sweep_stats_prefixes]

      # total activity statistics
      sweep_stats_tot_act = []
      for sp in sweep_stats_prefixes:
        for component in components:
          for tracker in component.get_tracking_vars():
            resource_list = np.sort(list(component.get_resources()))
            for resource in resource_list:
              sweep_stats_tot_activity = self.namingTemplates['tot_activity'].format(stats=sp, component=component.name, tracker=tracker, resource=resource)
              sweep_stats_tot_act.append(sweep_stats_tot_activity)













      for sweep_name in sweep_default + sweep_stats_tot_act:
        if sweep_name not in default_stats + default_stats_tot_act:
          self._updateCommaSeperatedList(group_final_return, sweep_name, organize_economics=has_mult_metrics)
    # opt mode uses optimization variable if no other stats are given, this is handled below
    if (case.get_mode == 'opt') and (case.get_optimization_settings() is not None):
      new_metric_opt_results = self._build_opt_metric_out_name(case)
      if (new_metric_opt_results != 'missing') and (new_metric_opt_results not in group_final_return.text):
        # additional results statistics have been requested, add this metric if not already present
        self._updateCommaSeperatedList(group_final_return, new_metric_opt_results, position=0)

    # fill out PostProcessor nodes
    pp_node = template.find('Models').find(".//PostProcessor[@name='statistics']")
    # add default statistics
    result_statistics = case.get_result_statistics() # list of stats beyond default
    tot_act_vars = []
    for component in components:
      for tracker in component.get_tracking_vars():
        resource_list = np.sort(list(component.get_resources()))
        for resource in resource_list:
          tot_act_var = "TotalActivity"+ "__" +component.name + "__" + tracker + "__" + resource
          tot_act_vars.append(tot_act_var)
    for var in tot_act_vars:
      for stat, pref in zip(DEFAULT_STATS_NAMES, default_stats_prefixes):
        pp_node.append(xmlUtils.newNode(stat, text=var, attrib={'prefix': pref}))

    for em in econ_metrics + tot_act_vars:
      for stat, pref in zip(DEFAULT_STATS_NAMES, default_stats_prefixes):
        pp_node.append(xmlUtils.newNode(stat, text=em, attrib={'prefix': pref}))
      # add any user supplied statistics beyond defaults
      if any(stat not in DEFAULT_STATS_NAMES + default_stats_tot_act for stat in result_statistics):
        for raven_metric_name in result_statistics:
          if raven_metric_name not in DEFAULT_STATS_NAMES:
            prefix = self._get_stats_metrics_prefixes(case, [raven_metric_name], use_extra=False)[0]
            # add subnode to PostProcessor
            if raven_metric_name == 'percentile':
              # add percent attribute
              percent = result_statistics[raven_metric_name]
              if isinstance(percent, list):
                for p in percent:
                  pp_node.append(xmlUtils.newNode(raven_metric_name, text=em,
                                                  attrib={'prefix': prefix,
                                                          'percent': p}))
              else:

                pp_node.append(xmlUtils.newNode(raven_metric_name, text=em,
                                                attrib={'prefix': prefix,
                                                        'percent': percent}))
            elif raven_metric_name in ['valueAtRisk', 'expectedShortfall', 'sortinoRatio', 'gainLossRatio']:
              threshold = result_statistics[raven_metric_name]
              if isinstance(threshold, list):
                for t in threshold:
                  if not em.startswith("TotalActivity"):
                    pp_node.append(xmlUtils.newNode(raven_metric_name, text=em,
                                                  attrib={'prefix': prefix,
                                                          'threshold': t}))
              else:
                if not em.startswith("TotalActivity"):
                  pp_node.append(xmlUtils.newNode(raven_metric_name, text=em,
                                                attrib={'prefix': prefix,
                                                        'threshold': threshold}))
            else:
              if not em.startswith("TotalActivity"):
                pp_node.append(xmlUtils.newNode(raven_metric_name, text=em,
                                              attrib={'prefix': prefix}))
              if em.startswith("TotalActivity"):
                if prefix not in FINANCIAL_PREFIXES:
                  pp_node.append(xmlUtils.newNode(raven_metric_name, text=em,
                                              attrib={'prefix': prefix}))

      # if not specified, "sweep" mode has additional defaults
      elif case.get_mode() == 'sweep':
        sweep_stats_prefixes = self._get_stats_metrics_prefixes(case, SWEEP_DEFAULT_STATS_NAMES, use_extra=False)
        for em in econ_metrics:
          for stat, pref in zip(SWEEP_DEFAULT_STATS_NAMES, sweep_stats_prefixes):
            pp_node.append(xmlUtils.newNode(stat, text=em, attrib={'prefix': pref}))
        for var in tot_act_vars:
          for stat, pref in zip(SWEEP_DEFAULT_STATS_NAMES, sweep_stats_prefixes):
            pp_node.append(xmlUtils.newNode(stat, text=var, attrib={'prefix': pref}))
    # if not specified, "opt" mode is handled in _modify_inner_optimization_settings

  def _modify_inner_data_handling(self, template, case):
    """
      Modifies template to include data handling options
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    # inner to outer
    ## default is netCDF, and is how the templates are already set up
    if case.data_handling['inner_to_outer'] == 'csv':
      # change the output IOStep outstream to do CSV instead of database
      print(
      template.find('Steps').find(".//IOStep[@name='database']")
      )
      db = template.find('Steps').find('.//IOStep[@name="database"]').find('.//Output[@class="Databases"]')
      db.attrib.update({'class': 'OutStreams', 'type': 'Print'})
      # the database and outstream print have the same name, so don't need to change text of node

  ##### OTHER UTILS #####
  def _add_arma_to_ensemble(self, template, source):
    """
      Adds an ARMA to EnsembleModel evaluation
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, source, HERON Placeholder, information about ARMA to be used
      @ Out, None
    """
    # pre-locate some useful nodes
    ens = template.find('Models').findall('EnsembleModel')[0] # NOTE relies on position
    data_objs = template.find('DataObjects')
    # pre-write some useful names
    inp_name = self.namingTemplates['data object'].format(source=source.name, contents='placeholder')
    eval_name = self.namingTemplates['data object'].format(source=source.name, contents='samples')
    # variables that are outputs of the rom
    out_vars = source.get_variable()

    # model was added during adding the loading step, so no need to add it to Models block
    # we do need to add it to the ensemble model though
    new_model = self._assemblerNode('Model', 'Models', 'ROM', source.name)
    new_model.append(self._assemblerNode('Input', 'DataObjects', 'PointSet', inp_name))
    new_model.append(self._assemblerNode('TargetEvaluation', 'DataObjects', 'DataSet', eval_name))
    ens.append(new_model)

    # create the data objects
    deps = {self.__case.get_time_name(): out_vars,
            self.__case.get_year_name(): out_vars}
    if source.eval_mode == 'clustered':
      deps[self.namingTemplates['cluster_index']] = out_vars

    self._create_dataobject(data_objs, 'PointSet', inp_name, inputs=['scaling'])
    self._create_dataobject(data_objs, 'DataSet', eval_name,
                            inputs=['scaling'],
                            outputs=out_vars,
                            depends=deps)

    # add variables to dispatch input requirements
    ## before all else fails, use variable groups
    # find dispatch_in_time group
    for group in (g for g in template.find('VariableGroups') if (g.tag == 'Group')):
      if group.attrib['name'] == 'GRO_dispatch_in_Time':
        break
    else:
      raise RuntimeError
    for var in out_vars:
      self._updateCommaSeperatedList(group, var)

  @staticmethod
  def _create_dataobject(dataobjects, typ, name, inputs=None, outputs=None, depends=None):
    """
      Creates a data object candidate to go to base class
      @ In, dataobjects, xml.etree.ElementTreeElement, DataObjects node
      @ In, typ, str, type of data object
      @ In, name, str, name of data object
      @ In, inputs, list(str), optional, input variable names
      @ In, outputs, list(str), optional, output variable names
      @ In, depends, dict, optional, time-dependency as {index: [stuff that depends on index]}
      @ Out, None
    """
    assert typ in ['PointSet', 'HistorySet', 'DataSet']
    new = xmlUtils.newNode(typ, attrib={'name':name})
    if inputs is not None:
      new.append(xmlUtils.newNode('Input', text=','.join(inputs)))
    if outputs is not None:
      new.append(xmlUtils.newNode('Output', text=','.join(outputs)))
    # index dependence
    ## if pointset, you went wrong somewhere if you gave dependencies
    if typ == 'PointSet':
      assert depends is None
    ## if a history set, there better only be one
    elif typ == 'HistorySet':
      assert depends is not None
      assert len(depends) == 1, f'Depends is: {depends}'
      opt = xmlUtils.newNode('options')
      opt.append(xmlUtils.newNode('pivotParameter', text=list(depends.keys())[0]))
      new.append(opt)
    ## otherwise, if dataset, follow the dependencies
    elif typ == 'DataSet':
      if depends is not None:
        for index, dep in depends.items():
          assert isinstance(dep, list)
          new.append(xmlUtils.newNode('Index', attrib={'var':index}, text=', '.join(dep)))
    dataobjects.append(new)

  @staticmethod
  def _find_cashflows(components):
    """
      Loop through comps and collect all the full cashflow names
      @ In, components, list, list of HERON Component instances for this run
      @ Out, cfs, list, list of cashflow full names e.g. {comp}_{cf}_CashFlow
    """
    cfs = []
    for comp in components:
      comp_name = comp.name
      for cashflow in comp.get_cashflows():
        cf_name = cashflow.name
        name = f'{comp_name}_{cf_name}_CashFlow'
        cfs.append(name)
        if cashflow._depreciate is not None:
          cfs.append(f'{comp_name}_{cf_name}_depreciation')
          cfs.append(f'{comp_name}_{cf_name}_depreciation_tax_credit')
    return cfs

  def _iostep_load_rom(self, template, case, components, source):
    """
      for INNER, creates new IOStep for loading a ROM
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    rom_name = source.name
    rom_source = source._target_file
    # add the step itself
    step_name = self.namingTemplates['stepname'].format(action='read', subject=rom_name)
    new_step = xmlUtils.newNode('IOStep', attrib={'name': step_name})
    new_step.append(self._assemblerNode('Input', 'Files', '', rom_source))
    new_step.append(self._assemblerNode('Output', 'Models', 'ROM', rom_name))
    template.find('Steps').append(new_step)
    # update the sequence
    self._updateCommaSeperatedList(template.find('RunInfo').find('Sequence'), new_step.attrib['name'], position=0)
    # add the model
    model = xmlUtils.newNode('ROM', attrib={'name':rom_name, 'subType':'pickledROM'})
    econ_comps = list(comp.get_economics() for comp in components)
    econ_global_params = case.get_econ(econ_comps) # FIXME: is this part still necessary?
    econ_global_settings = CashFlows.GlobalSettings()
    econ_global_settings.setParams(econ_global_params)
    ## update the ARMA model to sample a number of years equal to the ProjectLife from CashFlow
    if source.needs_multiyear is not None:
      multiyear = xmlUtils.newNode('Multicycle')
      multiyear.append(xmlUtils.newNode('cycles', text=source.needs_multiyear))
      model.append(multiyear)
    if source.limit_interp is not None:
      maxCycles = model.find('maxCycles')
      if maxCycles is not None:
        maxCycles.text = source.limit_interp
      else:
        model.append(xmlUtils.newNode('maxCycles', text=source.limit_interp))
    # change eval mode?
    if source.eval_mode == 'clustered':
      model.append(xmlUtils.newNode('clusterEvalMode', text='clustered'))
    template.find('Models').append(model)
    # add a file
    ## NOTE: the '..' assumes there is a working dir that is not ".", which should always be true.
    ## ALSO NOTE: the path to the ARMA file should now be absolute, so no directory fiddling necessary?
    template.find('Files').append(xmlUtils.newNode('Input', attrib={'name':rom_source}, text=rom_source)) # '../../'+rom_source
    # done

  def _iostep_rom_meta(self, template, source):
    """
      for INNER, create an IOStep for printing the ROM meta
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, source, HERON Placeholder, instance to add rom meta use for
      @ Out, None
    """
    rom_name = source.name
    # create the output data object
    objs = template.find('DataObjects')
    obj_name = f'{rom_name}_meta'
    self._create_dataobject(objs, 'DataSet', obj_name)
    # create the output outstream
    os_name = obj_name
    streams = template.find('OutStreams')
    if streams is None:
      streams = xmlUtils.newNode('OutStreams')
      template.append(streams)
    new = xmlUtils.newNode('Print', attrib={'name': os_name})
    new.append(xmlUtils.newNode('type', text='csv'))
    new.append(xmlUtils.newNode('source', text=obj_name))
    streams.append(new)
    # create the step
    step_name = self.namingTemplates['stepname'].format(action='print_meta', subject=rom_name)
    new_step = xmlUtils.newNode('IOStep', attrib={'name': step_name})
    new_step.append(self._assemblerNode('Input', 'Models', 'ROM', rom_name))
    new_step.append(self._assemblerNode('Output', 'DataObjects', 'PointSet', obj_name))
    new_step.append(self._assemblerNode('Output', 'OutStreams', 'Print', os_name))
    template.find('Steps').append(new_step)
    self._updateCommaSeperatedList(template.find('RunInfo').find('Sequence'), step_name, position=1)

  def _updateCommaSeperatedList(self, node, new, position=None, before=None, after=None, organize_economics=False):
    """
      Overloaded method from parent with new flag for multiple metrics.
      Parent: Statefully adds an entry to the given node's comma-seperated text
      If the node's text is empty, will insert "new" as the sole text (no commas).
      @ In, node, xml.etree.ElementTree.Element, node whose text is a comma-seperated string list
      @ In, new, str, name of entry to add
      @ In, position, int, optional, index where new should be inserted in sequence
      @ In, before, str, optional, entry name before which new should be added
      @ In, after, str, optional, entry name after which new should be added
      @ In, organize_economics, bool, optional, are we expecting multiple metrics?
      @ Out, None
    """
    # if we are expecting multiple metrics, they may be out of order - this block helps order them
    if organize_economics:
      # all entries with full name and with JUST the economic metric (e.g., NPV, IRR) respectively
      entries = list(x.strip() for x in node.text.split(',')) if node.text is not None else []
      #metric_entries = list(x.split('_')[-1] for x in entries)
      metric_entries = []
      for x in entries:
        if x.startswith("perc_") or x.startswith("glr_") or x.startswith("sortino_") or x.startswith("es_") or x.startswith("VaR_"):
           metric_entry = x.split("_", 2)[2]
        else:
          metric_entry = x.split("_", 1)[1]
        metric_entries.append(metric_entry)

      # getting the economic metric name, the index where it is first found and counts
      metric, ind_start, ind_len = np.unique(metric_entries, return_index=True, return_counts=True)
      # index map (e.g., {'NPV':(0,3), }) to get first instance of metric in list + # of times found
      metric_ind_map = {a:(b,c) for a,(b,c) in zip(metric, zip(ind_start, ind_len))}

      if new.startswith("perc_") or new.startswith("sortino_") or new.startswith("es_") or new.startswith("glr_") or new.startswith("VaR_"):
        first, length = metric_ind_map[new.split("_", 2)[2]]
      else:
        first, length = metric_ind_map[new.split("_", 1)[1]]


      #first, length = metric_ind_map[new.split('_')[-1]]
      # finding appropriate relative location within entries list to add new entry
      if first == 0:
        before = entries[length]
      else:
        after = entries[int(first+length-1)]
    # calling parent method
    super()._updateCommaSeperatedList(node, new, position, before, after)

  @staticmethod
  def _remove_by_name(root, removable):
    """
      Removes subs of "root" whose "name" attribute is in "removable"
      @ In, root. ET.Element, node whose subs should be searched through
      @ In, removable, list(str), names to remove
      @ Out, None
    """
    to_remove = []
    for node in root:
      if node.get('name', None) in removable:
        to_remove.append(node)
    for node in to_remove:
      root.remove(node)

  @staticmethod
  def _build_opt_metric_out_name(case):
    """
      Constructs the output name of the metric specified as the optimization objective
      @ In, case, HERON Case, defining Case instance
      @ Out, opt_out_metric_name, str, output metric name for use in inner/outer files
    """
    try:
      # metric name in RAVEN
      optimization_settings = case.get_optimization_settings()
      metric_raven_name = optimization_settings['stats_metric']['name']
      # potential metric name to add
      opt_out_metric_name = case.stats_metrics_meta[metric_raven_name]['prefix']
      # do I need to add a percent or threshold to this name?
      if metric_raven_name == 'percentile':
        opt_out_metric_name += '_' + str(optimization_settings['stats_metric']['percent'])
      elif metric_raven_name in ['valueAtRisk', 'expectedShortfall', 'sortinoRatio', 'gainLossRatio']:
        opt_out_metric_name += '_' + str(optimization_settings['stats_metric']['threshold'])
      opt_econ_metric, _ = case.get_opt_metric()
      output_econ_metric_name = case.economic_metrics_meta[opt_econ_metric]['output_name']
      opt_out_metric_name += f'_{output_econ_metric_name}'
    except (TypeError, KeyError):
      # <optimization_settings> node not in input file OR
      # 'metric' is missing from _optimization_settings
      opt_out_metric_name = 'missing'

    return opt_out_metric_name

  @staticmethod
  def _build_result_statistic_names(case, components):
    """
      Constructs the names of the statistics requested for output
      @ In, case, HERON Case, defining Case instance
      @ Out, names, list, list of names of statistics requested for output
    """
    names = []
    econ_metrics = case.get_econ_metrics(nametype='output')
    result_statistics = case.get_result_statistics()

    tot_act_vars = []
    for component in components:
      for tracker in component.get_tracking_vars():
        resource_list = np.sort(list(component.get_resources()))
        for resource in resource_list:
          tot_act_var = "TotalActivity__" + component.name + "__" + tracker + "__"+ resource
          tot_act_vars.append(tot_act_var)

    for e_metric in econ_metrics+tot_act_vars:
      for name in result_statistics:
        out_name = case.stats_metrics_meta[name]['prefix']
        # do I need to add percent or threshold?
        if name in ['percentile', 'valueAtRisk', 'expectedShortfall', 'sortinoRatio', 'gainLossRatio']:
          # multiple percents or thresholds may be specified
          if isinstance(result_statistics[name], list):
            for attrib in result_statistics[name]:
              names.append(out_name+'_'+attrib+'_'+e_metric)
          else:
            names.append(out_name+'_'+result_statistics[name]+'_'+e_metric)
        else:
          out_name += '_'+e_metric
          names.append(out_name)
    removed_names = [] # removed some names because it does not make sense to calculate financial metrics of total activity
    for name in names:
      if "TotalActivity" in name:
        if name.split("_")[0] in FINANCIAL_PREFIXES:
          removed_names.append(name)
    names = list(set(names) - set(removed_names))
    return names

  @staticmethod
  def _get_stats_metrics_prefixes(case, stats_names, use_extra=True):
    """
      Constructs the prefixes of the statistics requested for output
      @ In, case, HERON Case, defining Case instance
      @ In, stats_names, list, list of names of statistics requested for application to econ metrics
      @ Out, output, list, list of prefixes of statistics requested for output
    """
    output_names = []
    for name in stats_names:
      # we are assuming here that all stats metrics are applicable to all econ metrics.
      # revisit if this changes (e.g., we disallow "VaR" for IRR)
      output = case.stats_metrics_meta[name]['prefix']
      if name == "percentile" and use_extra:
        for perc in case.stats_metrics_meta[name]['percent']:
          n_output = f"{output}_{perc}"
          output_names.append(n_output)
      else:
        output_names.append(output)
    return output_names


# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Holds the template information for creating LCOE SWEEP OPT input files.
"""
import os
import sys
import copy
import shutil
import time
import xml.etree.ElementTree as ET
import itertools as it

import numpy as np
import dill as pk
import yaml

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

class TemplateAbce(TemplateBase, Base):
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
                                   'data object'    : '{source}_{contents}',
                                   'distribution'   : '{unit}_{feature}_dist',
                                   'ARMA sampler'   : '{rom}_sampler',
                                   'lib file'       : 'heron.lib', # TODO use case name?
                                   'cashfname'      : '_{component}{cashname}',
                                   're_cash'        : '_rec_{period}_{driverType}{driverName}',
                                   'cluster_index'  : '_ROM_Cluster',
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
  # def __repr__(self):
  #   """
  #     String representation of this Handler and its VP
  #     @ In, None
  #     @ Out, repr, str, string representation
  #   """
  #   msg = f'<Template Driver>'
  #   return msg

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Base.__init__(self, **kwargs)
    here = os.path.dirname(os.path.abspath(sys.modules[self.__class__.__module__].__file__))
    self._template_path = here
    self._template_outer_path = None
    self._template_abce_inputs_path = None
    self._template_abce_path = None
    self._template_outer = None
    self._template_abce_settings = None
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
    self._template_outer_path = os.path.join(rel_path, 'outer.xml')
    self._template_outer, _ = xmlUtils.loadToTree(self._template_outer_path, preserveComments=True)


  def createWorkflow(self, case, components, sources):
    """
      Create workflow XMLs
      @ In, case, HERON case, case instance for this sim
      @ In, components, list, HERON component instances for this sim
      @ In, sources, list, HERON source instances for this sim
      @ Out, outer, XML Element, root node for outer

    """
    # store pieces
    self.__case = case
    self.__components = components
    self.__sources = sources
    dispatcher_settings = case.dispatcher._disp_settings
    self._template_abce_inputs_path = dispatcher_settings['inputs_path']
    self._template_abce_path = dispatcher_settings['location']
    self._template_abce_settings = dispatcher_settings['settings_file']
    # initialize case economics
    case.load_econ(components)
    # load a copy of the template
    outer = copy.deepcopy(self._template_outer)
    # modify the templates
    outer = self._modify_outer(outer, case, components, sources)
    # create a list of abce input files to copy 
    # files should be in the self._template_abce_inputs_path
    # and not be a folder
    abce_files_to_copy = []
    for file in os.listdir(self._template_abce_inputs_path):
      if os.path.isfile(os.path.join(self._template_abce_inputs_path, file)):
        abce_files_to_copy.append(os.path.join(self._template_abce_inputs_path, file))
    ts_files = [os.path.join(self._template_abce_inputs_path, 'ts_data', file) for file in os.listdir(os.path.join(self._template_abce_inputs_path, 'ts_data'))]

    abce_files_to_copy.append(self._template_abce_settings)

    print('abce_files_to_copy',abce_files_to_copy)
    print('ts_files',ts_files)
    return abce_files_to_copy,outer

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
    outer = templates
    outer_file = os.path.abspath(os.path.join(destination, 'outer.xml'))
    abce_settings_file = self._template_abce_settings

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
    default_stats = [f'mean_{case._metric}', f'std_{case._metric}', f'med_{case._metric}']
    for stat in default_stats:
      self._updateCommaSeperatedList(group_outer_results, stat)
    # make sure user provided statistics beyond defaults get there
    if any(stat not in ['expectedValue', 'sigma', 'median'] for stat in case._result_statistics):
      stats_list = self._build_result_statistic_names(case)
      for stat_name in stats_list:
        if stat_name not in default_stats:
          self._updateCommaSeperatedList(group_outer_results, stat_name)
    # sweep mode has default variable names
    elif case.get_mode() == 'sweep':
      sweep_default = [f'mean_{case._metric}', f'std_{case._metric}', f'med_{case._metric}', f'max_{case._metric}',
                       f'min_{case._metric}', f'perc_5_{case._metric}', f'perc_95_{case._metric}',
                       f'samp_{case._metric}', f'var_{case._metric}']
      for sweep_name in sweep_default:
        if sweep_name not in default_stats:
          self._updateCommaSeperatedList(group_outer_results, sweep_name)
    # opt mode adds optimization variable if not already there
    if (case.get_mode() == 'opt') and (case._optimization_settings is not None):
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
          for resource in component.get_resources():
            var_name = self.namingTemplates['dispatch'].format(component=name, tracker=tracker, resource=resource)
            self._updateCommaSeperatedList(group, var_name)

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
    

    settings_file = xmlUtils.newNode('Input', attrib={'name': "settings.yml", 'type' : ""}, text='settings.yml')
    files.append(settings_file)
    # check the settings file settings.yml for node file_paths
    # if it exists, then add the files to the list of files to be transferred
    with open(self._template_abce_settings) as f:
      settings = yaml.safe_load(f)
    # creat a subdirectory for the inputs
    if 'file_paths' in settings:
      for file, path in settings['file_paths'].items():
        file_node = xmlUtils.newNode('Input', attrib={'name': file, 
                                                      'type' : "", 
                                                      'subDirectory': "inputs"}, text=path)
        files.append(file_node)
    files.remove(files.find('Input[@name="heron_lib"]'))
    files.remove(files.find('Input[@name="ABCE_sysimage_file"]'))
    files.remove(files.find('Input[@name="db_file"]'))
    files.remove(files.find('Input[@name="output_file"]'))
    files.remove(files.find('Input[@name="logo"]'))
    files.remove(files.find('Input[@name="inner_workflow"]'))
    # add the timeseries file to the list of files to be transferred
    # all the TS data should be in the ts_data folder under self._template_abce_inputs_path 
    # list the csv files in the directory
    # add them to the list of files to be transferred
    ts_data_path = os.path.join(self._template_abce_inputs_path, 'ts_data')
    for file in os.listdir(ts_data_path):
      if file.endswith('.csv'):
        file_node = xmlUtils.newNode('Input', attrib={'name': file, 
                                                      'type': "", 
                                                      'subDirectory': "inputs/ts_data"}, text=file)
        files.append(file_node)
    files.remove(files.find('Input[@name="repDays_35.csv"]'))

  def _modify_outer_models(self, template, case, components, sources):
    """
      Defines modifications to the Models of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    models = template.find('Models')
    raven = template.find('Models').find('Code')
    models.remove(raven)
    abce_gc = xmlUtils.newNode('code', attrib={'name': "abce", 'subType' : "GenericCode"})
    models.append(abce_gc)
    abce_exec = xmlUtils.newNode('executable', text=self._template_abce_path)
    prepend = xmlUtils.newNode('clargs', attrib={'arg': "python",'type' : "prepend"})
    settings_file = xmlUtils.newNode('clargs', attrib={'arg': "--settings_file", 
                                                       'extension':"yml",
                                                       'type' : "input",
                                                        'delimiter': "="})
    inputs_path = xmlUtils.newNode('clargs', attrib={'arg': "--inputs_path=inputs --verbosity=3", 
                                                     'type' : "text"})

    abce_gc.append(abce_exec)
    abce_gc.append(prepend)
    abce_gc.append(settings_file)
    abce_gc.append(inputs_path)

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
      self._remove_by_name(OSs, ['opt_path'])
    elif case.get_mode() == 'opt':
      self._remove_by_name(OSs, ['sweep'])
      # update plot 'opt_path' if necessary
      new_opt_objective = self._build_opt_metric_out_name(case)
      opt_path_plot_vars = OSs.find(".//Plot[@name='opt_path']").find('vars')
      if (new_opt_objective != 'missing') and (new_opt_objective not in opt_path_plot_vars.text):
        opt_path_plot_vars.text = opt_path_plot_vars.text.replace(f'mean_{case._metric}', new_opt_objective)
    # debug mode
    if case.debug['enabled']:
      # modify normal metric output
      out = OSs.findall('Print')[0]
      out.attrib['name'] = 'dispatch_print'
      out.find('source').text = 'dispatch'
      # handle dispatch plots for debug mode
      if case.debug['dispatch_plot']:
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
    if (case.get_mode() == 'opt') and (case._optimization_settings is not None) and (not case.debug['enabled']):  # TODO there should be a better way to handle the debug case
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
        type_node.text = case._optimization_settings['type']
      except KeyError:
        # type was not provided, so use the default value
        metric_raven_name = case._optimization_settings['metric']['name']
        type_node.text = case.metrics_mapping[metric_raven_name]['optimization_default']

      # swap out convergence values (only persistence implemented now)
      convergence = opt_node.find('convergence')
      persistence_node = convergence.find('persistence')
      try:
        persistence_node.text = str(case._optimization_settings['persistence'])
      except KeyError:
        # persistence was not provided, so use the default value
        pass

      # update convergence criteria, adding nodes as necessary
      convergence_settings = case._optimization_settings.get('convergence', {})
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
    multi_run_step = steps.findall('MultiRun')[0]
    for step in multi_run_step.findall('Input'):
      if step.text in ['inner_workflow','heron_lib']:
        multi_run_step.remove(step)

    model = multi_run_step.find('Model')
    model.text = 'abce'
    files = template.find('Files')
    for file in files:
      if file.tag == 'Input':
        file_node = xmlUtils.newNode('Input', attrib={'class': "Files", 
                                                      'type' : ""}, text=file.attrib['name'])
        multi_run_step.append(file_node)
    # add files 
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
      sweep.append(self._assemblerNode('Output', 'Databases', 'NetCDF', 'dispatch'))
      # add an output step to print/plot summaries
      io_step = ET.SubElement(steps, 'IOStep', attrib={'name': 'debug_output'})
      io_input = ET.SubElement(io_step, 'Input', attrib={'class': 'DataObjects', 'type': 'DataSet'})
      io_input.text = 'dispatch'
      io_step.append(self._assemblerNode('Output', 'OutStreams', 'Print', 'dispatch_print'))
      if case.debug['dispatch_plot']:
        io_output = ET.SubElement(io_step, 'Output', attrib={'class': 'OutStreams', 'type': 'Plot'})
        io_output.text = 'dispatchPlot'

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

  ##### OTHER UTILS #####

  def _remove_by_name(self, root, removable):
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

  # def _build_opt_metric_out_name(self, case):
  #   """
  #     Constructs the output name of the metric specified as the optimization objective
  #     @ In, case, HERON Case, defining Case instance
  #     @ Out, opt_out_metric_name, str, output metric name for use in inner/outer files
  #   """
  #   try:
  #     # metric name in RAVEN
  #     metric_raven_name = case._optimization_settings['metric']['name']
  #     # potential metric name to add
  #     opt_out_metric_name = case.metrics_mapping[metric_raven_name]['prefix']
  #     # do I need to add a percent or threshold to this name?
  #     if metric_raven_name == 'percentile':
  #       opt_out_metric_name += '_' + str(case._optimization_settings['metric']['percent'])
  #     elif metric_raven_name in ['valueAtRisk', 'expectedShortfall', 'sortinoRatio', 'gainLossRatio']:
  #       opt_out_metric_name += '_' + str(case._optimization_settings['metric']['threshold'])
  #     opt_out_metric_name += '_'+case._metric
  #   except (TypeError, KeyError):
  #     # <optimization_settings> node not in input file OR
  #     # 'metric' is missing from _optimization_settings
  #     opt_out_metric_name = 'missing'

  #   return opt_out_metric_name

  # def _build_result_statistic_names(self, case):
  #   """
  #     Constructs the names of the statistics requested for output
  #     @ In, case, HERON Case, defining Case instance
  #     @ Out, names, list, list of names of statistics requested for output
  #   """
  #   names = []
  #   for name in case._result_statistics:
  #     out_name = case.metrics_mapping[name]['prefix']
  #     # do I need to add percent or threshold?
  #     if name in ['percentile', 'valueAtRisk', 'expectedShortfall', 'sortinoRatio', 'gainLossRatio']:
  #       # multiple percents or thresholds may be specified
  #       if isinstance(case._result_statistics[name], list):
  #         for attrib in case._result_statistics[name]:
  #           names.append(out_name+'_'+attrib+'_'+case._metric)
  #       else:
  #         names.append(out_name+'_'+case._result_statistics[name]+'_'+case._metric)
  #     else:
  #       out_name += '_'+case._metric
  #       names.append(out_name)

  #   return names


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

import numpy as np
import dill as pk

# load utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from base import Base
import _utils as hutils
sys.path.pop()

# get raven location
RAVEN_LOC = hutils.get_raven_loc()
CF_LOC = hutils.get_cashflow_loc(raven_path=RAVEN_LOC)
if CF_LOC is None:
  raise RuntimeError('TEAL has not been found!\n' +
                     f'Check TEAL installation for the RAVEN at "{RAVEN_LOC}"')

sys.path.append(os.path.join(CF_LOC, '..'))
from TEAL.src.main import getProjectLength
from TEAL.src import CashFlows
sys.path.pop()

sys.path.append(os.path.join(RAVEN_LOC, '..'))
from utils import xmlUtils
from InputTemplates.TemplateBaseClass import Template as TemplateBase
sys.path.pop()

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
  #var_template.append(xmlUtils.newNode('grid', attrib={'type':'value', 'construction':'custom'}))

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
    cash = self._modify_cash(cash, case, components, sources)
    return inner, outer, cash

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
    inner, outer, cash = templates
    outer_file = os.path.abspath(os.path.join(destination, 'outer.xml'))
    inner_file = os.path.abspath(os.path.join(destination, 'inner.xml'))
    cash_file = os.path.abspath(os.path.join(destination, 'cash.xml'))
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
    with open(cash_file, 'w') as f:
      f.write(xmlUtils.prettify(cash))
    self.raiseAMessage(msg_format.format(*os.path.split(cash_file)))
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
    self._modify_outer_models(template, case, components)
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
    elif case._mode == 'opt':
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
      # for now, outer does not use InternalParallel
      batchSize = run_info.find('batchSize')
      batchSize.text = f'{case.outerParallel}'
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
    caps.text = ', '.join(f'{x.name}_capacity' for x in components if (x.get_capacity(None, raw=True).type not in ['Function', 'SyntheticHistory']))
    # outer results
    if case._optimization_settings is not None:
      group_outer_results = var_groups.find(".//Group[@name='GRO_outer_results']")
      new_metric_outer_results = self._build_opt_metric_out_name(case)
      if (new_metric_outer_results != 'missing') and (new_metric_outer_results not in group_outer_results.text):
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
      # -> synthetic histories?
      group = var_groups.find(".//Group[@name='GRO_outer_debug_synthetics']")
      for source in sources:
        if source.is_type('ARMA'):
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
    # update optimization settings if provided
    if (case.get_mode() == 'opt') and (case._optimization_settings is not None):
      new_opt_objective = self._build_opt_metric_out_name(case)
      # check if the metric in 'opt_eval' needs to be changed
      opt_eval_output_node = DOs.find(".//PointSet[@name='opt_eval']").find('Output')
      if (new_opt_objective != 'missing') and (new_opt_objective != opt_eval_output_node.text):
        opt_eval_output_node.text = new_opt_objective
      # check if the metric in 'opt_soln' needs to be changed
      opt_soln_output = DOs.find(".//PointSet[@name='opt_soln']").find('Output')
      if (new_opt_objective != 'missing') and (new_opt_objective not in opt_soln_output.text):
        # remove mean_NPV and replace with new_opt_objective
        opt_soln_output.text = opt_soln_output.text.replace('mean_NPV', new_opt_objective)
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

  def _modify_outer_models(self, template, case, components):
    """
      Defines modifications to the Models of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ Out, None
    """
    raven = template.find('Models').find('Code')
    # executable
    raven_exec = raven.find('executable')
    raven_exec.text = os.path.abspath(os.path.join(RAVEN_LOC, '..', 'raven_framework'))
    # conversion script
    conv = raven.find('conversion').find('input')
    conv.attrib['source'] = '../write_inner.py'
    # aliases
    text = 'Samplers|MonteCarlo@name:mc_arma_dispatch|constant@name:{}_capacity'
    for component in components:
      name = component.name
      attribs = {'variable':'{}_capacity'.format(name), 'type':'input'}
      new = xmlUtils.newNode('alias', text=text.format(name), attrib=attribs)
      raven.append(new)
    # if debug, grab the dispatch output instead of the summary
    if case.debug['enabled']:
      raven.find('outputDatabase').text = 'disp_full'


    # label aliases placed inside models
    text = 'Samplers|MonteCarlo@name:mc_arma_dispatch|constant@name:{}_label'
    for var, _ in self.__case.get_labels().items():
      attribs = {'variable': '{}_label'.format(var), 'type':'input'}
      new = xmlUtils.newNode('alias', text=text.format(var), attrib=attribs)
      raven.append(new)

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
      samps_node = template.find('Samplers').find('Grid')
    else:
      samps_node = template.find('Optimizers').find('GradientDescent')
    if case.debug['enabled']:
      samps_node.tag = 'MonteCarlo'
      samps_node.attrib['name'] = 'mc'
      init = xmlUtils.newNode('samplerInit')
      init.append(xmlUtils.newNode('limit', text=1))
      samps_node.append(init)
    # number of denoisings
    ## assumption: first node is the denoises node
    samps_node.find('constant').text = str(case.get_num_samples())
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
    print()

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

  def _modify_outer_optimizers(self, template, case):
    """
      Defines modifications to the Optimizers of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """

    # only modify if optimization_settings is in Case
    if (case.get_mode() == 'opt') and (case._optimization_settings is not None):
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
        type_node.text = case.optimization_metrics_mapping[metric_raven_name]['default']

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
    dist_name = self.namingTemplates['distribution'].format(unit=comp_name, feature='capacity')
    dist = copy.deepcopy(self.dist_template)
    dist.attrib['name'] = dist_name
    min_cap = min(capacities)
    max_cap = max(capacities)
    dist.find('lowerBound').text = str(min_cap)
    dist.find('upperBound').text = str(max_cap)
    xml = copy.deepcopy(self.var_template)
    xml.attrib['name'] = var_name
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
    ext_model.set('ModuleToLoad', input_filepath)
    self._modify_inner_runinfo(template, case)
    self._modify_inner_sources(template, case, components, sources)
    # NOTE: this HAS to come before modify_inner_denoisings,
    #       because we'll be copy-pasting these for each denoising --> or wait, maybe that's for the Outer to do!
    self._modify_inner_components(template, case, components)
    self._modify_inner_caselabels(template, case)
    self._modify_inner_time_vars(template, case)
    self._modify_inner_optimization_settings(template, case)
    if case.debug['enabled']:
      self._modify_inner_debug(template, case, components)
    # TODO modify based on resources ... should only need if units produce multiple things, right?
    # TODO modify CashFlow input ... this will be a big undertaking with changes to the inner.
    ## Maybe let the user change them? but then we don't control the variable names. We probably have to do it.
    return template

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
      groups[tag] = var_groups.find(".//Group[@name='GRO_{}']".format(tag))
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
      elif capacity.type in ['SyntheticHistory', 'Function', 'Variable']:
        # capacity is limited by a signal, so it has to be handled in the dispatch; don't include it here.
        # OR capacity is limited by a function, and we also can't handle it here, but in the dispatch.
        pass
      else:
        raise NotImplementedError('Capacity from "{}" not implemented yet. Component: {}'.format(capacity, cap_name))

      for tracker in component.get_tracking_vars():
        for resource in component.get_resources():
          var_name = self.namingTemplates['dispatch'].format(component=name, tracker=tracker, resource=resource)
          self._updateCommaSeperatedList(groups['init_disp'], var_name)
          self._updateCommaSeperatedList(groups['full_dispatch'], var_name)

  def _modify_inner_debug(self, template, case, components):
    """
      Modify template to work in a debug mode.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ Out, None
    """
    # RunInfo
    seq = template.find('.//RunInfo/Sequence')
    seq.text = (','.join(seq.text.split(',')[:-2])) # exclude basic stats parts
    # Steps
    for step in template.find('Steps'):
      if step.get('name') == 'arma_sampling':
        step.append(self._assemblerNode('Output', 'DataObjects', 'DataSet', 'disp_full'))
      # elif step.get('name') == 'write_summary':
      #   step.find('Output').text = 'disp_full'
    # Model
    extmod_vars = template.find('Models').find('ExternalModel').find('variables')
    self._updateCommaSeperatedList(extmod_vars, 'GRO_full_dispatch')
    self._updateCommaSeperatedList(extmod_vars, 'GRO_full_dispatch_indices')
    # DataObject
    datasets = template.find('DataObjects').findall('DataSet')
    for ds in datasets:
      if ds.attrib['name'] == 'dispatch_eval':
        break
    else:
      raise RuntimeError
    ds.append(xmlUtils.newNode('Output', text='GRO_full_dispatch'))
    for idx in ds.findall('Index'):
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
    if (case.get_mode() == 'opt') and (case._optimization_settings is not None):
      # optimization objective name provided (or 'missing')
      new_objective = self._build_opt_metric_out_name(case)
      # add optimization objective name to VariableGroups 'GRO_final_return' if not already there
      group = template.find('VariableGroups').find(".//Group[@name='GRO_final_return']")
      if (new_objective != 'missing') and (new_objective not in group.text):
        self._updateCommaSeperatedList(group, new_objective, position=0)
      # add optimization objective to PostProcessor list if not already there
      if new_objective != 'missing':
        pp_node = template.find('Models').find(".//PostProcessor[@name='statistics']")
        raven_metric_name = case._optimization_settings['metric']['name']
        prefix = case.optimization_metrics_mapping[raven_metric_name]['prefix']
        if pp_node.find(raven_metric_name) is None:
          # add subnode to PostProcessor
          if 'threshold' in case._optimization_settings['metric'].keys():
            if raven_metric_name in ['valueAtRisk', 'expectedShortfall']:
              threshold = str(case._optimization_settings['metric']['threshold'])
            else:
              threshold = case._optimization_settings['metric']['threshold']
              # TODO should NPV be the only metric available?
            new_node = xmlUtils.newNode(raven_metric_name, text='NPV',
                                        attrib={'prefix': prefix,
                                                'threshold': threshold})
          else:
            # TODO should NPV be the only metric available?
            new_node = xmlUtils.newNode(raven_metric_name, text='NPV',
                                        attrib={'prefix': prefix})
          pp_node.append(new_node)
        else:
          # check that subnode has correct values
          subnode = pp_node.find(raven_metric_name)
          # check that prefix is correct
          if prefix != subnode.attrib['prefix']:
            subnode.attrib['prefix'] = prefix
          # percentile has additional parameter to check
          if 'percent' in case._optimization_settings['metric'].keys():
            # defaults to 5 or 95 percentile
            if str(int(case._optimization_settings['metric']['percent'])) not in ['5', '95']:
              # update attribute
              subnode.attrib['percent'] = str(case._optimization_settings['metric']['percent'])


  ##### CASHFLOW #####
  def _modify_cash(self, template, case, components, sources):
    """
      Defines modifications to the cash.xml extension to RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, template, xml.etree.ElementTree.Element, modified template
    """
    self._modify_cash_Global(template, case)
    self._modify_cash_components(template, case, components)
    return template

  def _modify_cash_Global(self, template, case):
    """
      Defines modifications to Global of cash.xml extension to RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    # load variables
    tax = case._global_econ['tax']
    verbosity = case._global_econ['verbosity']
    inflation = case._global_econ['inflation']
    indicator = case._global_econ['Indicator']
    discountRate = case._global_econ['DiscountRate']
    projectTime = case._global_econ.get('ProjectTime', None)
    # set variables
    template.attrib['verbosity'] = str(verbosity)
    cash_global = template.find('Global')
    cash_global.find('DiscountRate').text = str(discountRate)
    cash_global.find('tax').text = str(tax)
    cash_global.find('inflation').text = str(inflation)
    cash_global.find('Indicator').attrib['name'] = indicator['name'][0]
    cash_global.find('Indicator').text = '\n      '.join(indicator['active'][:])
    if projectTime is not None:
      cash_global.append(xmlUtils.newNode('ProjectTime', text=str(projectTime)))

  def _modify_cash_components(self, template, case, components):
    """
      Defines modifications to Components of cash.xml extension to RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ Out, template, xml.etree.ElementTree.Element, modified template
    """
    for component in components:
      subComp = xmlUtils.newNode('Component', attrib={'name': component.name}, text='')
      subEconomics = component.get_economics()
      Life_time = subEconomics._lifetime
      subComp.append(xmlUtils.newNode('Life_time', text=str(Life_time)))
      cfs=xmlUtils.newNode('CashFlows')
      for subCash in subEconomics._cash_flows:
        driverName  = self.namingTemplates['cashfname'].format(component=subCash._component.name, cashname=subCash._driver.name)
        driverType   = subCash._driver.type

        inflation    = subCash._inflation
        mult_target  = subCash._mult_target
        name         = subCash.name
        tax          = subCash._taxable
        depreciation = subCash._depreciate
        if subCash._type == 'one-time':
          cfNode =  xmlUtils.newNode('Capex', text='', attrib={'name':'{name}'.format(name = name),
                                                                'tax':tax,
                                                                'inflation': inflation,
                                                                'mult_target': mult_target
                                                                })
          cfNode.append(xmlUtils.newNode('driver',text = driverName))
          cfNode.append(xmlUtils.newNode('alpha',text = subCash._alpha.get_value()))
          cfNode.append(xmlUtils.newNode('reference',text = subCash._reference.get_value()))
          cfNode.append(xmlUtils.newNode('X',text = subCash._scale.get_value()))
          if depreciation:
            cfNode.append(xmlUtils.newNode('depreciation',attrib={'scheme':'MACRS'}, text = depreciation))
          cfs.append(cfNode)
        else:
          cfNode =  xmlUtils.newNode('Recurring', text='', attrib={'name':'{name}'.format(name = name),
                                                                   'tax':tax,
                                                                   'inflation': inflation,
                                                                   'mult_target': mult_target
                                                                    })
          cfNode.append(xmlUtils.newNode('driver',
          text = self.namingTemplates['re_cash'].format(period=subCash._period,
                                                        driverType = driverType,
                                                        driverName ='_{comp}_{name}'.format(comp = component.name ,name = name))))

          cfNode.append(xmlUtils.newNode('alpha',text = '-1.0'))
          cfs.append(cfNode)
      subComp.append(cfs)
      template.append(subComp)



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

  def _create_dataobject(self, dataobjects, typ, name, inputs=None, outputs=None, depends=None):
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
      assert len(depends) == 1, 'Depends is: {}'.format(depends)
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
    econ_global_params = case.get_econ(econ_comps)
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
    obj_name = '{}_meta'.format(rom_name)
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

  def _build_opt_metric_out_name(self, case):
    """
      Constructs the output name of the metric specified as the optimization objective
      @ In, case, HERON Case, defining Case instance
      @ Out, opt_out_metric_name, str, output metric name for use in inner/outer files
    """
    try:
      # metric name in RAVEN
      metric_raven_name = case._optimization_settings['metric']['name']
      # potential metric name to add to VariableGroups, DataObjects, Optimizers
      opt_out_metric_name = case.optimization_metrics_mapping[metric_raven_name]['prefix']
      # do I need to add a percent or threshold to this name?
      if metric_raven_name == 'percentile':
        opt_out_metric_name += '_' + str(case._optimization_settings['metric']['percent'])
      elif metric_raven_name in ['valueAtRisk', 'expectedShortfall']:
        opt_out_metric_name += '_' + str(case._optimization_settings['metric']['threshold'])
      elif metric_raven_name in ['sortinoRatio', 'gainLossRatio']:
        opt_out_metric_name += '_' + case._optimization_settings['metric']['threshold']
      # add target variable to name TODO should this be changeable from NPV?
      opt_out_metric_name += '_NPV'
    except (TypeError, KeyError):
      # <optimization_settings> node not in input file OR
      # 'metric' is missing from _optimization_settings
      opt_out_metric_name = 'missing'

    return opt_out_metric_name

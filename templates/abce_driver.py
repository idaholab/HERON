
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
import io

import yaml
from ravenframework.MessageHandler import MessageHandler
# load utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from HERON.src.base import Base
import HERON.src._utils as hutils
from HERON.templates.template_driver import Template as HeronTemplate
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

class TemplateAbce(HeronTemplate, TemplateBase, Base):
  """
    Template for ABCE sweep class
    This template is designed to be used with the ABCE dispatcher. The workflow
    copies a set of input files from the ABCE input directory, then replaces
    the values in the input files with the values from the HERON inputs. 
    
    The ABCE input files are then run using the RAVEN sampler.
  """

  ############
  # API      #
  ############

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
    self._abce_files_to_copy = None
    self._abce_values_to_replace = {}
    self.__case = None
    self.__components = None
    self.__sources = None
    self.__sweep_vars = []
    self._working_dir = None
    self.messageHandler = MessageHandler()

  def loadTemplate(self, path):
    """
      Loads RAVEN template files from source.
      @ In, path, str, relative path to templates
      @ Out, None
    """
    rel_path = os.path.join(self._template_path, path)
    self._template_outer_path = os.path.join(rel_path, 'outer.xml')
    self._template_outer, _ = xmlUtils.loadToTree(self._template_outer_path, preserveComments=True)

  def createWorkflow(self, case, components, sources, loc):
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
    case_name = self.namingTemplates['jobname'].format(case=case.name, io='o')
    # create the location if not already there in the current working directory
    # the subfolder is called case_name
    if not os.path.exists(case_name):
      os.makedirs(case_name)
    # load a copy of the outer template
    outer = copy.deepcopy(self._template_outer)
    # modify the outer templates
    outer = self._modify_outer(outer, case, components, sources)

    # get the abce_inputs files to copy
    self._abce_files_to_copy = self._get_abce_files_to_copy(outer, case, components, sources)
    abce_temp = self._abce_files_to_copy
    # copy the files to the location
    # location is the subfolder in the current working directory called case_name
    self._working_dir = os.path.join(loc, case_name)
    abce_inputs = self._copied_abce_inputs(self._working_dir, abce_temp)
    # modyify the abce_inputs
    self._modify_abce_inputs(abce_inputs, outer, case, components, sources)

    return outer
 
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
    # abce_settings_file = self._template_abce_settings

    self.raiseAMessage('========================')
    self.raiseAMessage('HERON: writing files ...')
    self.raiseAMessage('========================')

    msg_format = 'Wrote "{1}" to "{0}/"'
    with open(outer_file, 'w') as f:
      f.write(xmlUtils.prettify(outer))
    self.raiseAMessage(msg_format.format(*os.path.split(outer_file)))
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
      comp_cap = comp.get_capacity(None, raw=True)
      comp_cap_type = comp_cap.type
      if comp_cap_type  not in ['Function', 'ARMA', 'SyntheticHistory', 'StaticHistory']:
        if comp_cap_type == 'FixedValue':
          vals = comp_cap.get_value(debug=case.debug['enabled'])
          self._abce_values_to_replace[f'{comp.name}|capacity'] = vals
        else:
          var_list.append(f'{comp.name}_capacity')
          self.__sweep_vars.append(f'{comp.name}_capacity')
    caps.text = ', '.join(var_list)

    # outer results
    group_outer_results = var_groups.find(".//Group[@name='GRO_outer_results']")
    # add required defaults
    default_metrics_point = ['OutputPlaceHolder']
    group_outer_results.text = ', '.join(default_metrics_point)

    # add another group for abce if component have value for capex and OM
    var_groups.append(xmlUtils.newNode('Group', attrib={'name': 'GRO_abce_capex'}))
    group_abce_capex = var_groups.find(".//Group[@name='GRO_abce_capex']")
    capex_list = []
    var_groups.append(xmlUtils.newNode('Group', attrib={'name': 'GRO_abce_FOM'}))
    group_abce_fom = var_groups.find(".//Group[@name='GRO_abce_FOM']")
    fom_list = []
    var_groups.append(xmlUtils.newNode('Group', attrib={'name': 'GRO_abce_VOM'}))
    group_abce_vom = var_groups.find(".//Group[@name='GRO_abce_VOM']")
    vom_list = []
    var_groups.append(xmlUtils.newNode('Group', attrib={'name': 'GRO_abce_FC'}))
    group_abce_fc = var_groups.find(".//Group[@name='GRO_abce_FC']")
    fc_list = []

    for comp in components:
      comp_eco = comp.get_economics()
      comp_life = comp_eco.get_lifetime()
      self._abce_values_to_replace[f'{comp.name}|unit_life'] = comp_life
      cfs=comp_eco.get_cashflows()
      for cf in cfs:
        # if cf._alpha.type is not FixedValue added to the list 
        # else value_to_change needs to be updated with the key 
        # as the name of the component and cf.name and value as cf._alpha.get_value()
        if cf._alpha.type == 'FixedValue':
          if cf.name == 'capex':
            abce_name = f'{comp.name}|overnight_capital_cost'
          if cf.name == 'fixed_OM':
            abce_name = f'{comp.name}|FOM'
          if cf.name == 'var_OM':
            abce_name = f'{comp.name}|VOM'
          if cf.name == 'fuel_cost':
            abce_name = f'{comp.name}|FC_per_MMBTU'
          self._abce_values_to_replace[abce_name] = cf._alpha.get_value()
        else:
          if cf.name=='capex' and cf._type=='one-time':
            capex_list.append(f'{comp.name}_capex')
            self.__sweep_vars.append(f'{comp.name}_capex')
          if cf.name=='fixed_OM' and cf._type=='repeating':
            fom_list.append(f'{comp.name}_FOM')
            self.__sweep_vars.append(f'{comp.name}_FOM')
          if cf.name=='var_OM' and cf._type=='repeating':
            vom_list.append(f'{comp.name}_VOM')
            self.__sweep_vars.append(f'{comp.name}_VOM')
          if cf.name=='fuel_cost' and cf._type=='repeating':
            fc_list.append(f'{comp.name}_FC')
            self.__sweep_vars.append(f'{comp.name}_FC')
    group_abce_capex.text = ', '.join(capex_list)
    group_abce_fom.text = ', '.join(fom_list)
    group_abce_vom.text = ', '.join(vom_list)
    group_abce_fc.text = ', '.join(fc_list)
    # remove empty groups
    groups_to_remove = []
    for group in var_groups:
      if group.text is None or group.text == '':
        groups_to_remove.append(group.get('name'))
    self._remove_by_name(var_groups, groups_to_remove)

  def _modify_outer_dataobjects(self, template, case, components):
    """
      Defines modifications to the DataObjects of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ Out, None
    """
    DOs = template.find('DataObjects')
    # remove opt components if not used
    if case.get_mode() == 'sweep' or case.debug['enabled']:
      self._remove_by_name(DOs, ['opt_eval', 'opt_soln'])
    elif case.get_mode() == 'opt':
      self._remove_by_name(DOs, ['grid'])
    for dataObject in DOs:
      if dataObject.get('name') == 'grid':
        # add variable groups to the input
        var_groups = template.find('VariableGroups')
        for group in var_groups:
          # if group is not none and is do not have outer in name
          # add the group to the dataObject input
          if group.text is not None and 'outer' not in group.get('name'):
            group_name = group.get('name')
            input = dataObject.find('Input')
            # check if the input already has a group
            if group_name not in input.text:
              input.text = input.text + ', ' + group_name

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
    # create a subdirectory for the inputs
    if 'file_paths' in settings:
      for file, path in settings['file_paths'].items():
        file_node = xmlUtils.newNode('Input', attrib={'name': file, 
                                                      'type' : "", 
                                                      'subDirectory': "inputs"}, text=path)
        files.append(file_node)
    # TODO add C2N_project_definitions.yml here but it might not be needed for single agent runs
    files.append(xmlUtils.newNode('Input', attrib={'name': "C2N_project_definitions.yml", 'type' : "",'subDirectory': "inputs"}, text='C2N_project_definitions.yml'))

    # remove the files that are not needed
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
    # remove the file name start with repDays like repDays_35 or repDays_45 etc
    # this is because the number of days is not known at this point
    for file in files:
      if file.get('name').startswith('repDays'):
        files.remove(file)

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
    abce_gc = xmlUtils.newNode('Code', attrib={'name': "abce", 'subType' : "Abce"})
    models.append(abce_gc)
    abce_exec = xmlUtils.newNode('executable', text=self._template_abce_path)
    prepend = xmlUtils.newNode('clargs', attrib={'arg': "python",'type' : "prepend"})
    settings_file = xmlUtils.newNode('clargs', attrib={'arg': "--settings_file", 
                                                       'extension':".yml",
                                                       'type' : "input",
                                                        'delimiter': "="})
    inputs_path = xmlUtils.newNode('clargs', attrib={'arg': "--inputs_path=inputs --verbosity=3", 
                                                     'type' : "text"})

    abce_gc.append(abce_exec)
    abce_gc.append(prepend)
    abce_gc.append(settings_file)
    abce_gc.append(inputs_path)

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
      #remove denoises variable
      samps_node.remove(samps_node.find('.//constant[@name="denoises"]'))      
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
      # NOTE this algorithm does not check for everything to be swept! Future work could expand it.
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
          dist, xml = self._create_new_sweep_variable(name, var_name, vals, sampler)
          dists_node.append(dist)
          samps_node.append(xml)
          # NOTE assumption (input checked): only one interaction per component
        # if not being swept, then it's just a fixed value. Just change the value in the abce input
      else:
        # this capacity will be evaluated by ARMA/Function, and doesn't need to be added here.
        pass

    for component in components:
      name = component.name
      comp_eco = component.get_economics()
      cfs=comp_eco.get_cashflows()
      for cf in cfs:
        if cf.name == 'fixed_OM':
          feature_name='FOM'
        elif cf.name == 'var_OM':
          feature_name='VOM'
        elif cf.name == 'fuel_cost':  
          feature_name='FC'
        elif cf.name == 'capex':
          feature_name='capex'
        var_name = self.namingTemplates['variable'].format(unit=component.name, feature=feature_name)
        vals = cf._alpha.get_value(debug=case.debug['enabled'])
        if isinstance(vals, list):
          dist, xml = self._create_new_sweep_variable(name, var_name, vals, sampler)
          dists_node.append(dist)
          samps_node.append(xml)

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

  def _create_new_sweep_variable(self, comp_name, var_name, capacities, sampler):
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
    elif 'dispatch' in var_name:
      dist_name = self.namingTemplates['distribution'].format(unit=comp_name, feature='dispatch')
    elif 'capex' in var_name:
      dist_name = self.namingTemplates['distribution'].format(unit=comp_name, feature='capex')
    elif 'FOM' in var_name:
      dist_name = self.namingTemplates['distribution'].format(unit=comp_name, feature='fom')
    elif 'VOM' in var_name:
      dist_name = self.namingTemplates['distribution'].format(unit=comp_name, feature='vom')
    elif 'FC' in var_name:
      dist_name = self.namingTemplates['distribution'].format(unit=comp_name, feature='fc')

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

  ##### ABCE #####

  def _get_abce_files_to_copy(self,template,case, components, sources):
    """
      Defines files to copy from ABCE to HERON working directory.
      @ In, template, xml.etree.ElementTree.Element, template to modify.
      @ In, case, Case, HERON case object
      @ In, components, list, list of components
      @ In, sources, list, list of sources
      @ Out, files, list, list of files to copy
    """
    # create a list of abce input files to copy 
    # files should be in the self._template_abce_inputs_path
    # and not be a folder
    abce_files_to_copy = {}
    # add input files as a list
    abce_input_files = []

    files_in_outer = []
    for file in template.find('Files'):
      files_in_outer.append(file.text)

    for file in os.listdir(self._template_abce_inputs_path):
      if file in files_in_outer and os.path.isfile(os.path.join(self._template_abce_inputs_path, file)):
        self.raiseAMessage('Copying ABCE file "{}" from ABCE inputs folder to HERON'.format(file))
        abce_input_files.append(os.path.join(self._template_abce_inputs_path, file))
        
    ## ts_files are the files in DataGenerators

    ts_files = []
    for file in sources:
      ts_files.append(file._target_file)
    abce_files_to_copy['abce_input_files'] = abce_input_files
    abce_files_to_copy['ts_files'] = ts_files
    # add abce_settings file
    abce_files_to_copy['abce_settings_file'] = self._template_abce_settings
    return abce_files_to_copy

  def _copied_abce_inputs(self, loc, abce_temp):
    """
      Copy ABCE input files to working directory.
      @ In, loc, str, location to copy files to
      @ In, abce_temp, dict, dictionary of ABCE files to copy
      @ Out, abce_input_files_copied, dict, dictionary of ABCE files copied
    """
    # copy abce_settings file to working directory
    shutil.copy(abce_temp['abce_settings_file'], loc)
    # create inputs folder in working directory if not exist
    if not os.path.exists(os.path.join(loc, 'inputs')):
      os.makedirs(os.path.join(loc, 'inputs'))
    # copy abce input files to working directory
    for file in abce_temp['abce_input_files']:
      shutil.copy(file, os.path.join(loc, 'inputs'))
    # create ts_data folder under inputs folder in working directory if not exist
    if not os.path.exists(os.path.join(loc, 'inputs', 'ts_data')):
      os.makedirs(os.path.join(loc, 'inputs', 'ts_data'))
    # only copy csv files to working directory 
    for file in abce_temp['ts_files']:
      if file.endswith('.csv'):
        shutil.copy(file, os.path.join(loc, 'inputs', 'ts_data'))
    # create a dict of abce input files copied with relative path
    abce_input_files_copied = {}
    abce_input_files_copied['abce_settings_file'] = os.path.join(loc, os.path.basename(abce_temp['abce_settings_file']))
    abce_input_files_copied['abce_input_files'] = [os.path.join(loc, 'inputs', os.path.basename(file)) for file in abce_temp['abce_input_files']]
    abce_input_files_copied['ts_files'] = [os.path.join(loc, 'inputs', 'ts_data', os.path.basename(file)) for file in abce_temp['ts_files']]
    return  abce_input_files_copied

  def _modify_abce_inputs(self, abce_inputs, outer, case, components, sources):
    """
      Modifies ABCE input files to match HERON case.
      @ In, abce_inputs, dict, dictionary of ABCE files copied
      @ In, outer, xml.etree.ElementTree.Element, outer node of template
      @ In, case, HERON Case, defining Case instance
      @ In, components, dict, dictionary of components
      @ In, sources, dict, dictionary of sources
      @ Out, None
    """
    # Modify ABCE settings file
    self._modify_abce_settings(abce_inputs['abce_settings_file'], case, components, sources)
    
    # Modify ABCE input files
    other_inputs = abce_inputs['abce_input_files']
    self._modify_abce_other_input(other_inputs, outer, case, components, sources)
    
    # Modify time series data files
    for file in abce_inputs['ts_files']:
      self._modify_ts_data(file, case, components, sources)

  def _modify_abce_settings(self, abce_settings_file, case, components, sources):
    """
      Modifies ABCE settings file to match HERON case.
      @ In, abce_settings_file, str, ABCE settings file settings.yml
      @ In, case, HERON Case, defining Case instance
      @ In, components, dict, dictionary of components
      @ In, sources, dict, dictionary of sources
      @ Out, None
    """
    # Parse the YAML file
    with open(abce_settings_file, 'r') as f:
      settings = yaml.load(f, Loader=yaml.FullLoader)
    
    # Modify the YAML file
    yaml_lines = self._modify_abce_settings_yml(settings, case, components, sources)
    
    # Write the modified YAML back to the file
    with open(abce_settings_file, 'w') as f:
      f.write('\n'.join(yaml_lines))

  def _modify_abce_settings_yml(self, settings, case, components, sources):
    """
      Modifies ABCE settings yml file to match HERON case.
      @ In, settings, dict, ABCE settings dictionary
      @ In, case, HERON Case, defining Case instance
      @ In, components, dict, dictionary of components
      @ In, sources, dict, dictionary of sources
      @ Out, None
    """
    dispatch_settings = case.dispatcher._disp_settings
    # Update the settings dictionary
    for key, value in settings['dispatch'].items():
      if key in dispatch_settings:
        settings['dispatch'][key] = dispatch_settings[key]
    
    econ_settings = case._global_econ
    self._update_dict(settings['scenario'], econ_settings)
    yaml_lines = self._modify_abce_settings_global(settings)
    
    return yaml_lines

  def _update_dict(self, target, source):
    """
      Updates target dictionary with source dictionary.
      @ In, target, dict, target dictionary
      @ In, source, dict, source dictionary
      @ Out, None
    """
    for key, value in source.items():
      if isinstance(value, dict) and key in target and isinstance(target[key], dict):
        self._update_dict(target[key], value)
      elif isinstance(value, list) and key in target and isinstance(target[key], list):
        target[key] = value

  def _modify_abce_settings_global(self,yaml_data):
    """
      Modifies ABCE settings yml file to match HERON case.
      @ In, yaml_data, dict, dictionary of data in yml file
      @ Out, yaml_lines, list, list of lines in yml file
    """
    yaml_lines = []
    # modify the global settings
    for key, value in yaml_data.items():
      # if key starts with '_', remove it
      if key[0] == '_':
        key = key[1:]
      yaml_lines.append(f'{key}: ')
      for k, v in value.items(): 
        # add indentation for all the subkeys 
        stream = io.StringIO()
        # if v is a dict, add indentation for all the subkeys in v
        yaml.dump({k: v}, stream, indent=2) 
        # if stream has multiple lines, add indentation for all the lines, if it contains quotes, remove them
        if len(stream.getvalue().splitlines()) > 1:
          for line in stream.getvalue().splitlines():
              line = '  ' + line
              # if line.strip()[-1] is a quote, remove it
              if line.strip()[-1] == "'":
                yaml_lines.append(line.replace("'", ""))
              elif line.strip()[-1] == '"':
                yaml_lines.append(line.replace('"', ""))
              else:
                yaml_lines.append(line)
        else:
          # if stream.getvalue().strip() have quotes for the last word, remove them
          if stream.getvalue().strip()[-1] == '"':
            yaml_lines.append('  ' + stream.getvalue().strip().replace('"', ""))
          elif stream.getvalue().strip()[-1] == "'":
            yaml_lines.append('  ' + stream.getvalue().strip().replace("'", ""))
          else:
            yaml_lines.append('  ' + stream.getvalue().strip())
      yaml_lines.append('\n')
    

    for i, line in enumerate(yaml_lines):
      if line.strip().startswith('-'):
        line = '  ' + line
        yaml_lines[i] = line
    return yaml_lines

  def _modify_abce_other_input(self, abce_input_files, outer, case, components, sources):
    """
      Modifies ABCE input files to match HERON case.
      @ In, abce_input_files, dict, dictionary of ABCE input files
      @ In, outer, ElementTree, outer XML element
      @ In, case, HERON Case, defining Case instance
      @ In, components, dict, dictionary of components
      @ In, sources, dict, dictionary of sources
      @ Out, None
    """
    if not self.__sweep_vars and not self._abce_values_to_replace:
      self.raiseAMessage(f'No need to modify other ABCE input files, since no sweep variables are defined, and no ABCE values to replace are defined')
      return
      
    unit_specs = outer.find('Files').find(".//Input[@name='unit_specs_data_file']")
    sub_dir = unit_specs.attrib['subDirectory']
    unit_specs_file = os.path.join(self._working_dir, sub_dir, unit_specs.text)
      
    if self.__sweep_vars or self._abce_values_to_replace:
      # Locate the node in the unit_specs_file which is a YAML file
      # Read the YAML file
      with open(unit_specs_file, 'r') as f:
        unit_specs_data = yaml.safe_load(f)
          
      # Modify ABCE input files changing the values to RAVEN variables if needed
      if self._abce_values_to_replace:
        self.raiseAMessage(f'For ABCE input files, modifying the values to HERON input values')
        for value_to_change in self._abce_values_to_replace.items():
          self.raiseAMessage(f'Modifying {value_to_change[0]} to {value_to_change[1]}')
          # Split the alias
          unit_name, abce_node = value_to_change[0].split('|')
          unit = unit_specs_data[unit_name]
          # Find the alias in the unit
          for k, v in unit.items():
            if k == abce_node:
              # Replace the value
              unit[k] = value_to_change[1]
              unit_specs_data[unit_name] = unit
          
      if self.__sweep_vars:
        self.raiseAMessage(f'For ABCE input files, modifying the values to RAVEN variables')
        for raven_alias in self.__sweep_vars:
          self.raiseAMessage(f'Modifying {raven_alias}')
          # Split the alias from the last underscore
          abce_alias = raven_alias.split('_')[-1]
          if abce_alias == 'capex':
            abce_node = 'overnight_capital_cost'
          elif abce_alias == 'FC':
            abce_node = 'FC_per_MMBTU'
          else:
            abce_node = abce_alias
          # Unit name is the rest of the alias
          unit_name = raven_alias.replace(f'_{abce_alias}', '')
          # Find the unit in the unit_specs_data
          unit = unit_specs_data[unit_name]
          # Changed_alias is $RAVEN-raven_alias$
          changed_alias = f'$RAVEN-{raven_alias}$'
          # Replace the key name from abce_node to raven_alias
          unit[abce_node] = changed_alias
          # Replace the unit in the unit_specs_data
          unit_specs_data[unit_name] = unit
          
      # Write the unit_specs_data back to the unit_specs_file
      with open(unit_specs_file, 'w') as f:
        yaml.dump(unit_specs_data, f)

  def _modify_ts_data(self, ts_file, case, components, sources):
    """
      Modifies time series data file to match HERON case.
      @ In, ts_file, str, time series data file
      @ In, case, HERON Case, defining Case instance
      @ In, components, dict, dictionary of components
      @ In, sources, dict, dictionary of sources
      @ Out, None
    """
    # Currently, no modification is needed.
    # TODO: In the future, we may need to modify the ts data file to match HERON case for ARMA.
    pass

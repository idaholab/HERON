
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
## don't pop from path so we can use it later
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
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

class Template(TemplateBase):
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
                                   'dispatch'       : 'Dispatch__{component}__{resource}',
                                   'data object'    : '{source}_{contents}',
                                   'distribution'   : '{unit}_{feature}_dist',
                                   'ARMA sampler'   : '{rom}_sampler',
                                   'lib file'       : 'heron.lib', # TODO use case name?
                                   'cashfname'      : '_{component}{cashname}',
                                   're_cash'        : '_rec_{period}_{driverType}{driverName}'
                                  })

  # template nodes
  dist_template = xmlUtils.newNode('Uniform')
  dist_template.append(xmlUtils.newNode('lowerBound'))
  dist_template.append(xmlUtils.newNode('upperBound'))

  var_template = xmlUtils.newNode('variable')
  var_template.append(xmlUtils.newNode('distribution'))
  var_template.append(xmlUtils.newNode('grid', attrib={'type':'value', 'construction':'custom'}))

  ############
  # API      #
  ############
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    here = os.path.dirname(os.path.abspath(sys.modules[self.__class__.__module__].__file__))
    self._template_path = here
    self._template_inner_path = None
    self._template_outer_path = None
    self._template_cash_path = None
    self._template_cash = None
    self._template_inner = None
    self._template_outer = None

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
    # load a copy of the template
    inner = copy.deepcopy(self._template_inner)
    outer = copy.deepcopy(self._template_outer)
    cash = copy.deepcopy(self._template_cash)

    # modify the templates
    inner = self._modify_inner(inner, case, components, sources)
    outer = self._modify_outer(outer, case, components, sources)
    cash = self._modify_cash(cash, case, components, sources)
    # TODO write other files, like cashflow inputs?
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
    print('HERON: writing files ...')
    msg_format = ' ... wrote "{1:15s}" to "{0}/"'
    with open(outer_file, 'w') as f:
      f.write(xmlUtils.prettify(outer))
    print(msg_format.format(*os.path.split(outer_file)))
    with open(inner_file, 'w') as f:
      f.write(xmlUtils.prettify(inner))
    print(msg_format.format(*os.path.split(inner_file)))
    with open(cash_file, 'w') as f:
      f.write(xmlUtils.prettify(cash))
    print(msg_format.format(*os.path.split(cash_file)))
    # write library of info so it can be read in dispatch during inner run
    data = (self.__case, self.__components, self.__sources)
    lib_file = os.path.abspath(os.path.join(destination, self.namingTemplates['lib file']))
    with open(lib_file, 'wb') as lib:
      pk.dump(data, lib)
    print(msg_format.format(*os.path.split(lib_file)))
    # copy "write_inner.py", which has the denoising and capacity fixing algorithms
    conv_src = os.path.abspath(os.path.join(self._template_path, 'write_inner.py'))
    conv_file = os.path.abspath(os.path.join(destination, 'write_inner.py'))
    shutil.copyfile(conv_src, conv_file)
    print(msg_format.format(*os.path.split(conv_file)))
    # run, if requested
    if run:
      self.runWorkflow(destination)

  ############
  # UTILS    #
  ############
  ##### OUTER #####
  def _modify_outer(self, template, case, components, sources):
    """
      Defines modifications to the outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, template, xml.etree.ElementTree.Element, modified template
    """
    self._modify_outer_mode(template, case)
    self._modify_outer_runinfo(template, case)
    self._modify_outer_vargroups(template, components)
    self._modify_outer_files(template, sources)
    self._modify_outer_models(template, components)
    self._modify_outer_samplers(template, case, components)
    # TODO copy needed model/ARMA/etc files to Outer Working Dir so they're known
    # TODO including the heron library file
    return template

  def _modify_outer_mode(self, template, case):
    """
      Defines modifications throughout outer.xml RAVEN input file due to "sweep" or "opt" mode.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ Out, None
    """
    if case._mode == 'opt':
      # RunInfo
      template.find('RunInfo').find('Sequence').text = 'optimize'
      # Steps
      sweep = template.find('Steps').findall('MultiRun')[0]
      template.find('Steps').remove(sweep)
      # DataObjects
      grid = template.find('DataObjects').findall('PointSet')[0]
      template.find('DataObjects').remove(grid)
      # Samplers
      template.remove(template.find('Samplers'))
      # OutStreams
      sweep = template.find('OutStreams').findall('Print')[0]
      template.find('OutStreams').remove(sweep)
    else: # mode is 'sweep'
      # RunInfo
      template.find('RunInfo').find('Sequence').text = 'sweep'
      # Steps
      opt = template.find('Steps').findall('MultiRun')[1]
      template.find('Steps').remove(opt)
      # DataObjects
      opt_eval = template.find('DataObjects').findall('PointSet')[1]
      opt_soln = template.find('DataObjects').findall('PointSet')[2]
      template.find('DataObjects').remove(opt_eval)
      template.find('DataObjects').remove(opt_soln)
      # Optimizers
      template.remove(template.find('Optimizers'))
      # OutStreams
      opt_soln = template.find('OutStreams').findall('Print')[1]
      template.find('OutStreams').remove(opt_soln)

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

  def _modify_outer_vargroups(self, template, components):
    """
      Defines modifications to the VariableGroups of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, components, list, list of HERON Component instances for this run
      @ Out, None
    """
    var_groups = template.find('VariableGroups')
    # capacities
    caps = var_groups[0]
    caps.text = ', '.join('{}_capacity'.format(x.name) for x in components)

  def _modify_outer_files(self, template, sources):
    """
      Defines modifications to the Files of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, sources, list, list of HERON Placeholder instances for this run
      @ Out, None
    """
    files = template.find('Files')
    # modify path to inner
    inner = files.find('Input') # NOTE assuming it's the first file in the template
    inner.text = '../inner.xml'
    # add other files needed by inner (functions, armas, etc)
    for source in sources:
      if source.is_type('Function'):
        # add it to the list of things that have to be transferred
        files = template.find('Files')
        src = xmlUtils.newNode('Input', attrib={'name': 'transfers'}, text='../'+source._source)
        files.append(src)

  def _modify_outer_models(self, template, components):
    """
      Defines modifications to the Models of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
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

  def _modify_outer_samplers(self, template, case, components):
    """
      Defines modifications to the Samplers/Optimizers of outer.xml RAVEN input file.
      @ In, template, xml.etree.ElementTree.Element, root of XML to modify
      @ In, case, HERON Case, defining Case instance
      @ In, components, list, list of HERON Component instances for this run
      @ Out, None
    """
    """ TODO """
    dists_node = template.find('Distributions')
    if case._mode == 'sweep':
      samps_node = template.find('Samplers').find('Grid')
    else:
      samps_node = template.find('Optimizers').find('GradientDescent')
    # number of denoisings
    ## assumption: first node is the denoises node
    samps_node.find('constant').text = str(case._num_samples)
    # add sweep variables to input

    for component in components:
      interaction = component.get_interaction()
      # NOTE this algorithm does not check for everthing to be swept! Future work could expand it.
      ## Currently checked: Component.Interaction.Capacity
      ## --> this really needs to be made generic for all kinds of valued params!
      name = component.name
      var_name = self.namingTemplates['variable'].format(unit=name, feature='capacity')
      cap = interaction.get_capacity(None, None, None, None, raw=True)
      # do we already know the capacity values?
      if cap.type == 'value':
        vals = cap.get_values()
        # is the capacity variable being swept over?
        if isinstance(vals, list):
          # make new Distribution, Sampler.Grid.variable
          dist, for_grid, for_opt = self._create_new_sweep_capacity(name, var_name, vals)
          dists_node.append(dist)
          if case._mode == 'sweep':
            samps_node.append(for_grid)
          else:
            samps_node.append(for_opt)
          # NOTE assumption (input checked): only one interaction per component
        # if not being swept, then it's just a fixed value.
        else:
          samps_node.append(xmlUtils.newNode('constant', text=vals, attrib={'name': var_name}))
      else:
        # this capacity will be evaluated by ARMA/Function, and doesn't need to be added here.
        pass

  def _create_new_sweep_capacity(self, comp_name, var_name, capacities):
    """
      for OUTER, creates new distribution and variable for grid/opt sampling
      @ In, comp_name, str, name of component
      @ In, var_name, str, name of capacity variable
      @ In, capacities, list, float list of capacities to sweep/opt over
      @ Out, dist, xml.etree.ElementTree,Element, XML for distribution
      @ Out, grid, xml.etree.ElementTree,Element, XML for grid sampler variable
      @ Out, opt, xml.etree.ElementTree,Element, XML for optimizer variable
    """
    # distribution
    dist_name = self.namingTemplates['distribution'].format(unit=comp_name, feature='capacity')
    dist = copy.deepcopy(self.dist_template)
    dist.attrib['name'] = dist_name
    dist.find('lowerBound').text = str(min(capacities))
    dist.find('upperBound').text = str(max(capacities))
    # sampler variable, for Grid case
    grid = copy.deepcopy(self.var_template)
    grid.attrib['name'] = var_name
    grid.find('distribution').text = dist_name
    grid.find('grid').text = ' '.join(str(x) for x in sorted(capacities))
    # optimizer variable, for opt case
    opt = copy.deepcopy(grid)
    opt.remove(opt.find('grid'))
    initial = np.average(capacities)
    opt.append(xmlUtils.newNode('initial', text=initial))
    return dist, grid, opt

  ##### INNER #####
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
    # TODO modify based on resources ... should only need if units produce multiple things, right?
    # TODO modify CashFlow input ... this will be a big undertaking with changes to the inner.
    ## Maybe let the user change them? but then we don't control the variable names. We probably have to do it.
    return template

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
        # NOTE assuming input to all ARMAs is "scaling" constant = 1.0, already in MonteCarlo sampler
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
    for tag in ['capacities', 'init_disp', 'means']:
      groups[tag] = var_groups.find(".//Group[@name='GRO_{}']".format(tag))
    # change inner input due to components requested
    for component in components:
      name = component.name
      # treat capacity
      ## we just need to make sure everything we need gets into the dispatch ensemble model.
      ## For each interaction of each component, that means making sure the Function, ARMA, or constant makes it.
      ## Constants from outer (namely sweep/opt capacities) are set in the MC Sampler from the outer
      ## The Dispatch needs info from the Outer to know which capacity to use, so we can't pass it from here.
      interaction = component.get_interaction()
      capacity = interaction.get_capacity(None, None, None, None, raw=True)
      values = capacity.get_values()
      #cap_name = self.namingTemplates['variable'].format(unit=name, feature='capacity')
      if isinstance(values, (list, float)):

        # this capacity is being [swept or optimized in outer] (list) or is constant (float)
        # -> so add a node, put either the const value or a dummy in place
        cap_name = self.namingTemplates['variable'].format(unit=name, feature='capacity')
        if isinstance(values, list):
          cap_val = 42 # placeholder
        else:
          cap_val = values
        mc.append(xmlUtils.newNode('constant', attrib={'name': cap_name}, text=cap_val))
      elif values is None and capacity.type in ['ARMA', 'Function', 'variable']:
        # capacity is limited by a signal, so it has to be handled in the dispatch; don't include it here.
        # OR capacity is limited by a function, and we also can't handle it here, but in the dispatch.
        pass
      else:
        raise NotImplementedError('Capacity from "{}" not implemented yet. Component: {}'.format(capacity, cap_name))

      # add component to applicable variable groups
      self._updateCommaSeperatedList(groups['capacities'], cap_name)
      for resource in interaction.get_resources():
        var_name = self.namingTemplates['dispatch'].format(component=name, resource=resource)
        self._updateCommaSeperatedList(groups['init_disp'], var_name)
        self._updateCommaSeperatedList(groups['means'], var_name)

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
          cfNode.append(xmlUtils.newNode('alpha',text = subCash._alpha._value))
          cfNode.append(xmlUtils.newNode('reference',text = subCash._reference._value))
          cfNode.append(xmlUtils.newNode('X',text = subCash._scale._value))
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
    self._create_dataobject(data_objs, 'PointSet', inp_name, inputs=['scaling'])
    self._create_dataobject(data_objs, 'DataSet', eval_name,
                            inputs=['scaling'],
                            outputs=out_vars,
                            depends={'Time': out_vars, 'Year': out_vars}) # TODO user-defined?

    # add variables to dispatch input requirements
    ## before all else fails, use variable groups
    # find dispatch_in_time group
    for group in template.find('VariableGroups'):
      if group.attrib['name'] == 'GRO_dispatch_in_Time':
        break
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
    ## update the ARMA model to sample a number of years equal to the ProjectLife from CashFlow
    #comp_lifes = list(comp.get_economics().get_lifetime() for comp in components)
    #req_proj_life = case.get_econ().get('ProjectLife', None)
    econ_comps = list(comp.get_economics() for comp in components)
    econ_global_params = case.get_econ(econ_comps)
    econ_global_settings = CashFlows.GlobalSettings()
    econ_global_settings.setParams(econ_global_params)
    #project_life = getProjectLength(econ_global_settings, econ_comps) - 1 # skip construction year
    #multiyear = xmlUtils.newNode('Multiyear')
    #multiyear.append(xmlUtils.newNode('years', text=project_life))
    # TODO FIXME XXX growth param ????
    #model.append(multiyear)
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

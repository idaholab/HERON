"""
  Defines the Cases entity.
"""
from __future__ import unicode_literals, print_function
import os
import sys
import copy
import importlib

from base import Base
import Components
import Placeholders
raven_path = '~/projects/raven/raven_framework'
sys.path.append(os.path.expanduser(raven_path))
from utils import InputData, xmlUtils,InputTypes






class Case(Base):
  """
    Produces something, often as the cost of something else
    TODO this case is for "sweep-opt", need to make a superclass for generic
  """
  #### INITIALIZATION ####
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    #####
    """
    InputData Specification for the class Case.
    @ ModeOptions, minimize, maximize or sweep over multiple values of capacities
    @ EconMetrics, can be NPV (Net Present Value) and lcoe (levelized cost of energy)
    @ num_arma_samples, copies of the trained signals
    @ history_length, total length of the input ARMA signal
    @ ProjectTime, total length of the project
    @ DiscountRate, interest rate required to compute the econometrics
    @ tax, taxation rate
    @ inflation, inflation rate
    @ verbosity, length of the output argument
    @ resources, resource to be produced or consumed
    @ dispatch_increments, amount to be dispatched in a fixed time interval
    """

  

    #####
    input_specs = InputData.parameterInputFactory('Case', ordered=False, baseNode=None, descr= r""" The \xmlNode{Case} contains
    the basic parameters needed for a HERON case. """)
    input_specs.addParam('name', param_type=InputTypes.StringType, required=True, descr=r"""An appropriate user defined name of the case.""")

    mode_options = InputTypes.makeEnumType('ModeOptions', 'ModeOptionsType', ['min', 'max', 'sweep'])
    desc_mode_options = r""" Minimize, maximize or sweep over multiple values of capacities."""
    econ_metrics = InputTypes.makeEnumType('EconMetrics', 'EconMetricsTypes', ['NPV', 'lcoe'])
    desc_econ_metrics = r""" This metric can be NPV (Net Present Value) and lcoe (levelized cost of energy) used for techno-economic analysis of the power plants.""" 




    input_specs.addSub(InputData.parameterInputFactory('mode', contentType=mode_options,strictMode=True,
         descr=desc_mode_options))
    input_specs.addSub(InputData.parameterInputFactory('metric', contentType=econ_metrics, descr=desc_econ_metrics))
    input_specs.addSub(InputData.parameterInputFactory('differential', contentType=InputTypes.BoolType,strictMode=True,
         descr=r"""Differential represents the additional cashflow generated when building additional capacities.
        This value can be either \xmlString{True} or \xmlString{False}."""))
    input_specs.addSub(InputData.parameterInputFactory('num_arma_samples', contentType=InputTypes.IntegerType, descr=r"""Number of copies of the trained signals."""))
    input_specs.addSub(InputData.parameterInputFactory('timestep_interval', contentType=InputTypes.IntegerType, descr=r"""Time step interval between two values of signal."""))
    input_specs.addSub(InputData.parameterInputFactory('history_length', contentType=InputTypes.IntegerType, descr= r"""Total length of one realization of the ARMA signal."""))

    # economics global settings
    econ = InputData.parameterInputFactory('economics', ordered=False, descr= r"""\xmlNode{economics} contains the details of the econometrics
    computations to be performed by the code.""")
    econ.addSub(InputData.parameterInputFactory('ProjectTime', contentType=InputTypes.FloatType, descr=r"""Total length of the project."""))
    econ.addSub(InputData.parameterInputFactory('DiscountRate', contentType=InputTypes.FloatType, descr=r"""Interest rate required to compute the discounted cashflow (DCF)"""))
    econ.addSub(InputData.parameterInputFactory('tax', contentType=InputTypes.FloatType, descr= r"""Taxation rate is a metric which represents the 
    rate at which an individual or corporation is taxed."""))
    econ.addSub(InputData.parameterInputFactory('inflation', contentType=InputTypes.FloatType, descr=r"""Inflation rate is a metric which represents the
    the rate at which the average price level of a basket of selected goods and services in an economy increases over some period of time."""))
    econ.addSub(InputData.parameterInputFactory('verbosity', contentType=InputTypes.IntegerType, descr=r"""Length of the output argument."""))
    input_specs.addSub(econ)

    # increments for resources
    incr = InputData.parameterInputFactory('dispatch_increment', contentType=InputTypes.FloatType, descr=r"""This is the amount of resource to be dispatched in a fixed time interval.""")
    incr.addParam('resource', param_type=InputTypes.StringType, required=True, descr=r"""Resource to be consumed or produced.""")
    input_specs.addSub(incr)#, descr=r"""Resource to be produced or consumed""")

    return input_specs

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Base.__init__(self, **kwargs)
    self.name = None           # case name
    self._mode = None          # extrema to find: min, max, sweep
    self._metric = None        # economic metric to focus on: lcoe, profit, cost
    self._diff_study = None    # is this only a differential study?
    self._num_samples = 1      # number of ARMA stochastic samples to use ("denoises")
    self._hist_interval = None # time step interval, time between production points
    self._hist_len = None      # total history length, in same units as _hist_interval
    self._num_hist = None      # number of history steps, hist_len / hist_interval
    self._global_econ = {}     # global economics settings, as a pass-through
    self._increments = {} 
    self._Resample_T = None        # user-set increments for resources

  def read_input(self, xml):
    """
      Sets settings from input file
      @ In, xml, xml.etree.ElementTree.Element, input from user
      @ Out, None
    """
    # get specs for allowable inputs
    specs = self.get_input_specs()()
    specs.parseNode(xml)
    self.name = specs.parameterValues['name']
    for item in specs.subparts:
      if item.getName() == 'mode':
        self._mode = item.value
      elif item.getName() == 'metric':
        self._metric = item.value
      elif item.getName() == 'differential':
        self._diff_study = item.value
      elif item.getName() == 'num_arma_samples':
        self._num_samples = item.value
      elif item.getName() == 'Resample_T':
        self._Resample_T = item.value
      elif item.getName() == 'timestep_interval':
        self._hist_interval = float(item.value)
      elif item.getName() == 'history_length':
        self._hist_len = item.value
      elif item.getName() == 'economics':
        for sub in item.subparts:
          self._global_econ[sub.getName()] = sub.value
      elif item.getName() == 'dispatch_increment':
        self._increments[item.parameterValues['resource']] = item.value

    self._num_hist = self._hist_len // self._hist_interval # TODO what if it isn't even?
    self.raiseADebug('Successfully initialized Case {}.'.format(self.name))

  def __repr__(self):
    return '<HERON Case>'

  def print_me(self, tabs=0, tab='  '):
    """ Prints info about self """
    pre = tab*tabs
    print(pre+'Case:')
    print(pre+'  name:', self.name)
    print(pre+'  mode:', self._mode)
    print(pre+'  meric:', self._metric)
    print(pre+'  diff_study:', self._diff_study)

  #### ACCESSORS ####
  def get_increments(self):
    return self._increments

  def get_working_dir(self, which):
    if which == 'outer':
      io = 'o'
    elif which == 'inner':
      io = 'i'
    else:
      raise NotImplementedError('Unrecognized working dir request: "{}"'.format(which))
    return '{case}_{io}'.format(case=self.name, io=io)

  def get_econ(self, components):
    # only add additional params the first time this is called
    if 'active' not in self._global_econ:
      # NOTE self._metric can only be NPV right now! XXX TODO FIXME
      ## so no need for the "target" in the indicator
      indic = {'name': [self._metric]}
      indic['active'] = []
      for comp in components:
        comp_name = comp.name
        for cf in comp.get_cashflows():
          cf_name = cf.name
          indic['active'].append('{}|{}'.format(comp_name, cf_name))
      self._global_econ['Indicator'] = indic
    return self._global_econ

  def get_metric(self):
    return self._metric

  def get_mode(self):
    """ returns mode """
    return self._mode

  def get_num_samples(self):
    return self._num_samples

  def get_num_timesteps(self):
    return self._num_hist
#### ADDED ONE MORE ACCESSOR####
  def get_Resample_T(self):
    return self._Resample_T
  def get_hist_interval(self):
    return self._hist_interval
  
  def get_hist_length(self):
    return self._hist_len

  #### API ####

  def write_workflows(self, components, sources, loc):
    """
      Writes workflows for this case to XMLs on disk.
      @ In, components, HERON components, components for the simulation
      @ In, sources, HERON sources, sources for the simulation
      @ In, loc, str, location in which to write files
      @ Out, None
    """
    # load templates
    template_class = self._load_template()
    inner, outer, cash = template_class.createWorkflow(self, components, sources)

    template_class.writeWorkflow((inner, outer, cash), loc)

  #### UTILITIES ####
  def _load_template(self):
    """ TODO """
    src_dir = os.path.dirname(os.path.realpath(__file__))
    heron_dir = os.path.abspath(os.path.join(src_dir, '..'))
    template_dir = os.path.abspath(os.path.join(heron_dir, 'templates'))
    template_name = 'template_driver'
    # import template module
    sys.path.append(heron_dir)
    module = importlib.import_module('templates.{}'.format(template_name))
    # load template, perform actions
    template_class = module.Template()
    template_class.loadTemplate(None, template_dir)
    return template_class


  def _modify(self, templates, components, sources):
    """ TODO """
    outer = self._modify_outer(templates['outer'], components, sources)
    inner = self._modify_inner(templates['inner'], components, sources)
    return {'outer':outer, 'inner':inner}

  def _modify_outer(self, template, components, sources):
    """ TODO """
    ###################
    # RUN INFO        #
    ###################
    run_info = template.find('RunInfo')
    case_name = self.get_working_dir('outer') #self.string_templates['jobname'].format(case=self.name, io='o')
    ooooooooo # I don't think this gets run!
    # job name
    run_info.find('JobName').text = case_name
    # working dir
    run_info.find('WorkingDir').text = case_name
    # TODO sequence, maybe should be modified after STEPS (or part of it)

    ###################
    # STEPS           #
    # FIlES           #
    ###################
    # no ARMA steps, that's all on the inner

    ###################
    # VARIABLE GROUPS #
    ###################
    var_groups = template.find('VariableGroups')
    # capacities
    caps = var_groups[0]
    caps.text = ', '.join('{}_capacity'.format(x.name) for x in components)
    # results
    ## these don't need to be changed, they're fine.

    ###################
    # DATA OBJECTS    #
    ###################
    # thanks to variable groups, we don't really need to adjust these.

    ###################
    # MODELS          #
    ###################
    # aliases
    raven = template.find('Models').find('Code')
    text = 'Samplers|MonteCarlo@name:mc_arma|constant@name:{}_capacity'
    for component in components:
      name = component.name
      attribs = {'variable':'{}_capacity'.format(name), 'type':'input'}
      new = xmlUtils.newNode('alias', text=text.format(name), attrib=attribs)
      raven.append(new)
    # TODO location of RAVEN? We should know globally somehow.

    ###################
    # DISTRIBUTIONS   #
    # SAMPLERS        #
    ###################
    dists_node = template.find('Distributions')
    samps_node = template.find('Samplers').find('Grid')
    # number of denoisings
    ## assumption: first node is the denoises node
    samps_node.find('constant').text = str(self._num_samples)
    # add sweep variables to input
    dist_template = xmlUtils.newNode('Uniform')
    dist_template.append(xmlUtils.newNode('lowerBound'))
    dist_template.append(xmlUtils.newNode('upperBound'))
    var_template = xmlUtils.newNode('variable')
    var_template.append(xmlUtils.newNode('distribution'))
    var_template.append(xmlUtils.newNode('grid', attrib={'type':'value', 'construction':'custom'}))
    for component in components:
      interaction = component.get_interaction()
      # if produces, then its capacity might be flexible
      # TODO this algorithm does not check for everthing to be swept! Future work could expand it.
      ## Currently checked: Component.Interaction.Capacity
      # TODO doesn't test for Components.Demand; maybe the user wants to perturb the demand level?
      if isinstance(interaction, (Components.Producer, Components.Storage)):
        # is the capacity variable (being swept over)?
        name = component.name
        var_name = self.string_templates['variable'].format(unit=name, feature='capacity')
        if isinstance(interaction._capacity, list):
          dist_name = self.string_templates['distribution'].format(unit=name, feature='capacity')
          new = copy.deepcopy(dist_template)
          new.attrib['name'] = dist_name
          new.find('lowerBound').text = str(min(interaction._capacity))
          new.find('upperBound').text = str(max(interaction._capacity))
          dists_node.append(new)
          # also mess with the Sampler block
          new = copy.deepcopy(var_template)
          new.attrib['name'] = var_name
          new.find('distribution').text = dist_name
          new.find('grid').text = ', '.join(str(x) for x in sorted(interaction._capacity))
          samps_node.append(new)
          # TODO assumption (input checked): only one interaction per component
        # elif the capacity is fixed, it becomes a constant
        else:
          samps_node.append(xmlUtils.newNode('constant', text=interaction._capacity, attrib={'name': var_name}))

    ###################
    # OUTSTREAMS      #
    ###################
    # no changes needed here!

    # TODO copy needed model/ARMA/etc files to Outer Working Dir so they're known
    return template

  def _modify_inner(self, template, components, sources):
    """ TODO """
    ###################
    # RUN INFO        #
    # STEPS           #
    ###################
    run_info = template.find('RunInfo')
    steps = template.find('Steps')
    models = template.find('Models')
    # case, working dir
    case_name = self.get_working_dir('inner') #self.string_templates['jobname'].format(case=self.name, io='i')
    run_info.find('JobName').text = case_name
    run_info.find('WorkingDir').text = case_name
    # steps
    ## for now, just load all sources
    ## TODO someday, only load what's needed
    for source in sources:
      name = source.name
      if isinstance(source, Placeholders.ARMA):
        # add a model block
        models.append(xmlUtils.newNode('ROM', attrib={'name':name, 'subType':'pickledROM'}))
        # add a read step
        read_step_name = self.string_templates['stepname'].format(action='read', subject=name)
        new_step = xmlUtils.newNode('IOStep', attrib={'name': read_step_name})
        new_step.append(xmlUtils.newNode('Input', attrib={'class':'Files', 'type':''}, text=source._source))
        new_step.append(xmlUtils.newNode('Output', attrib={'class':'Models', 'type':'ROM'}, text=name))
        # add a sample step (sample #denoises)
        # add a custom sampler to read the samples
        ## sequence # NOTE order matters!
        seq = run_info.find('Sequence').text
        if seq is None:
          run_info.find('Sequence').text = read_step_name
        else:
          run_info.find('Sequence').text = '{}, {}'.format(seq, read_step_name)

        # TODO ensemble model? data object?
        steps.append(new_step)
      elif isinstance(source, Placeholders.CSV):
        # add a File
        # add a step to read
        # add a data object to read into
        ## TODO no one needs this right now, so forget it.
        pass # TODO
      elif isinstance(source, Placeholders.Function):
        pass # TODO




    return template








